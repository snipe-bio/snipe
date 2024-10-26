import concurrent.futures
import logging
import multiprocessing
import os
import re
import signal
import sys
import threading
import queue
from typing import Any, Dict, List, Optional, Tuple
from pyfastx import Fastx as SequenceReader # pylint: disable=no-name-in-module
import sourmash
from pathos.multiprocessing import ProcessingPool as Pool



class SnipeSketch:
    """
    SnipeSketch is responsible for creating FracMinHash sketches from genomic data.
    It supports parallel processing, progress monitoring, and different sketching modes
    including sample, genome, and amplicon sketching.
    """

    def __init__(self, enable_logging: bool) -> None:
        """
        Initialize the SnipeSketch instance.

        Args:
            enable_logging (bool): Flag to enable or disable logging.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self._configure_logging(enable_logging)

    def _configure_logging(self, enable_logging: bool) -> None:
        """
        Configure the logging for the class.

        Args:
            enable_logging (bool): Flag to enable or disable logging.
        """
        if enable_logging:
            self.logger.setLevel(logging.DEBUG)
            if not self.logger.hasHandlers():
                handler = logging.StreamHandler()
                handler.setLevel(logging.DEBUG)
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
            self.logger.debug("Logging is enabled for SnipeSketch.")
        else:
            self.logger.setLevel(logging.CRITICAL)

    # *******************************
    # *        Sketching            *
    # *******************************

    def process_sequences(
        self,
        fasta_file: str,
        thread_id: int,
        total_threads: int,
        progress_queue: multiprocessing.Queue,
        batch_size: int = 100_000,
        ksize: int = 51,
        scaled: int = 10_000,
    ) -> sourmash.MinHash:
        """
        Process a subset of sequences to create a FracMinHash sketch.

        Each process creates its own MinHash instance and processes sequences
        assigned based on the thread ID. Progress is reported via a shared queue.

        Args:
            fasta_file (str): Path to the FASTA file.
            thread_id (int): Identifier for the current thread.
            total_threads (int): Total number of threads.
            progress_queue (multiprocessing.Queue): Queue for reporting progress.
            batch_size (int, optional): Number of sequences per progress update. Defaults to 100_000.
            ksize (int, optional): K-mer size. Defaults to 51.
            scaled (int, optional): Scaling factor for MinHash. Defaults to 10_000.

        Returns:
            sourmash.MinHash: The resulting FracMinHash sketch.
        """
        self._register_signal_handler()
        try:
            fa_reader = SequenceReader(fasta_file)
            mh = sourmash.MinHash(
                n=0, ksize=ksize, scaled=scaled, track_abundance=True
            )
            local_count = 0
            base_count = 0

            for idx, (_, seq) in enumerate(fa_reader):
                if idx % total_threads == thread_id:
                    mh.add_sequence(seq, force=True)
                    local_count += 1
                    base_count += len(seq)

                    if local_count >= batch_size:
                        progress_queue.put(batch_size)
                        local_count = 0

            if local_count > 0:
                progress_queue.put(local_count)

            self.logger.debug(
                "Thread %d processed %d hashes.", thread_id, len(mh)
            )
            return mh, base_count

        except KeyboardInterrupt:
            self.logger.warning("KeyboardInterrupt detected in process_sequences.")
            sys.exit(0)
        except Exception as e:
            self.logger.error("Error in process_sequences: %s", e)
            raise

    def _register_signal_handler(self) -> None:
        """
        Register the signal handler for graceful shutdown.
        """
        signal.signal(signal.SIGINT, self._worker_signal_handler)

    def progress_monitor(
        self,
        progress_queue: multiprocessing.Queue,
        progress_interval: int,
        total_threads: int,
        stop_event: threading.Event,
    ) -> None:
        """
        Monitor and display the progress of sequence processing.

        Args:
            progress_queue (multiprocessing.Queue): Queue for receiving progress updates.
            progress_interval (int): Interval for progress updates.
            total_threads (int): Number of processing threads.
            stop_event (threading.Event): Event to signal the monitor to stop.
        """
        total = 0
        next_update = progress_interval
        try:
            while not stop_event.is_set() or not progress_queue.empty():
                try:
                    count = progress_queue.get(timeout=0.5)
                    total += count
                    if total >= next_update:
                        print(f"\rProcessed {next_update:,} sequences.", end="", flush=True)
                        next_update += progress_interval
                except queue.Empty:
                    continue
        except Exception as e:
            self.logger.error("Error in progress_monitor: %s", e)
        finally:
            print(f"\rProcessed {total:,} sequences in total.")

    def _worker_signal_handler(self, signum: int, frame: Any) -> None:
        """
        Handle signals in worker processes to exit gracefully.

        Args:
            signum (int): Signal number.
            frame (Any): Current stack frame.
        """
        self.logger.info("Received signal %d. Exiting worker.", signum)
        sys.exit(0)

    def _sketch_sample(
        self,
        sample_name: str,
        fasta_file: str,
        num_processes: int = 4,
        progress_interval: int = 1_000_000,
        batch_size: int = 100_000,
        k_size: int = 51,
        scale: int = 10_000,
        **kwargs: Any,
    ) -> Tuple[sourmash.SourmashSignature, int]:
        """
        Create a FracMinHash sketch for a sample using parallel processing.

        Args:
            sample_name (str): Name of the sample.
            fasta_file (str): Path to the FASTA file.
            num_processes (int, optional): Number of parallel processes. Defaults to 4.
            progress_interval (int, optional): Interval for progress updates. Defaults to 1_000_000.
            batch_size (int, optional): Number of sequences per progress update. Defaults to 100_000.
            k_size (int, optional): K-mer size. Defaults to 51.
            scale (int, optional): Scaling factor for MinHash. Defaults to 10_000.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Tuple[sourmash.SourmashSignature, int]: The resulting signature and total bases processed.
        """
        self.logger.info("Starting sketching with %d processes...", num_processes)

        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()
        stop_event = threading.Event()

        monitor_thread = threading.Thread(
            target=self.progress_monitor,
            args=(progress_queue, progress_interval, num_processes, stop_event),
            daemon=True,
        )
        monitor_thread.start()

        pool = Pool(nodes=num_processes)
        results: List[Any] = []

        try:
            for thread_id in range(num_processes):
                result = pool.apipe(
                    self.process_sequences,
                    fasta_file,
                    thread_id,
                    num_processes,
                    progress_queue,
                    batch_size,
                    k_size,
                    scale,
                )
                results.append(result)

            pool.close()
            pool.join()

        except KeyboardInterrupt:
            self.logger.warning("Interrupt received. Terminating processes...")
            pool.terminate()
            pool.join()
            stop_event.set()
            monitor_thread.join()
            sys.exit(1)

        except Exception as e:
            self.logger.error("Error during sketching: %s", e)
            pool.terminate()
            pool.join()
            stop_event.set()
            monitor_thread.join()
            raise

        finally:
            stop_event.set()
            monitor_thread.join()

        minhashes = []
        total_bases = 0
        
        for idx, result in enumerate(results):
            try:
                mh, base_count = result.get()
                if mh:
                    minhashes.append(mh)
                    total_bases += base_count
                    self.logger.debug("MinHash from thread %d collected with %d bases.", idx, base_count)
            except Exception as e:
                self.logger.error("Error retrieving MinHash from thread %d: %s", idx, e)

        if not minhashes:
            raise ValueError("No MinHashes were generated.")

        # Merge all MinHashes into one
        mh_full = minhashes[0]
        for mh in minhashes[1:]:
            mh_full.merge(mh)
            
        # append number of bases to the signature name for QC purposes
        sample_name += f";snipe_bases={total_bases}"

        signature = sourmash.SourmashSignature(mh_full, name=sample_name)
        self.logger.info("Sketching completed for sample: %s with total bases: %d", sample_name, total_bases)

        return signature, total_bases

    def sample_sketch(
        self,
        sample_name: str,
        filename: str,
        num_processes: int,
        batch_size: int,
        ksize: int,
        scale: int,
        **kwargs: Any,
    ) -> sourmash.SourmashSignature:
        """
        Generate a sketch for a sample and return its signature.

        Args:
            sample_name (str): Name of the sample.
            filename (str): Path to the input FASTA file.
            num_processes (int): Number of processes to use.
            batch_size (int): Batch size for processing.
            ksize (int): K-mer size.
            scale (int): Scaling factor.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            sourmash.SourmashSignature: The generated signature.

        Raises:
            RuntimeError: If an error occurs during sketching.
        """
        self.logger.info("Starting sample sketch for: %s", sample_name)
        try:
            signature, total_bases = self._sketch_sample(
                sample_name=sample_name,
                fasta_file=filename,
                num_processes=num_processes,
                batch_size=batch_size,
                k_size=ksize,
                scale=scale,
                **kwargs,
            )
            self.logger.info("Sample sketch completed for: %s, with total number of %d bases", sample_name, total_bases)
            return signature, total_bases
        except Exception as e:
            self.logger.error(
                "Error occurred during sample sketching: %s", str(e)
            )
            raise RuntimeError("Error occurred during sample sketching.") from e

    # *******************************
    # *      Genome Sketching       *
    # *******************************

    def parse_fasta_header(self, header: str) -> Tuple[str, str]:
        """
        Parse a FASTA header and categorize the sequence type.

        Args:
            header (str): The FASTA header string.

        Returns:
            Tuple[str, str]: A tuple containing the sequence type and name.
        """
        full_header = header.strip()
        header_lower = full_header.lower()

        if header_lower.startswith(">"):
            header_lower = header_lower[1:]
            full_header = full_header[1:]

        seq_type = "unknown"
        name = "unknown"

        patterns = {
            "scaffold": re.compile(r"\b(scaffold|unplaced|unlocalized)\b"),
            "contig": re.compile(r"\bcontig\b"),
            "mitochondrial DNA": re.compile(r"\b(mt|mitochondrion|mitochondrial|mitochondria|mito|mtdna)\b"),
            "chloroplast DNA": re.compile(r"\b(chloroplast|cpdna|plastid)\b"),
            "plasmid": re.compile(r"\bplasmid\b"),
            "chromosome": re.compile(r"\bchromosome\b|\bchr\b"),
            "reference chromosome": re.compile(r"NC_\d{6}\.\d+"),
        }

        for stype, pattern in patterns.items():
            if pattern.search(header_lower):
                if stype in {"scaffold", "contig", "plasmid"}:
                    match = re.match(r"(\S+)", full_header)
                    name = match.group(1) if match else "unknown"
                elif stype in {"mitochondrial DNA", "chloroplast DNA"}:
                    name = stype.split()[0]
                    stype = name.lower()
                elif stype == "chromosome":
                    match = re.search(r"(?:chromosome|chr)[_\s]*([^\s,]+)", header_lower)
                    if match:
                        name = match.group(1).rstrip(".,")
                        if name.upper() in {"X", "Y", "W", "Z"}:
                            stype = "sex"
                        elif name.upper() == "M":
                            stype = "mitochondrial"
                        else:
                            stype = "autosome"
                elif stype == "reference chromosome":
                    match = pattern.search(full_header)
                    if match and not (patterns["scaffold"].search(header_lower) or patterns["contig"].search(header_lower)):
                        name = match.group()
                return stype, name

        return seq_type, name

    def parallel_genome_sketching(
        self,
        fasta_file: str,
        cores: int = 1,
        ksize: int = 51,
        scale: int = 10_000,
        assigned_genome_name: str = "full_genome",
        **kwargs: Any,
    ) -> Tuple[sourmash.SourmashSignature, Dict[str, sourmash.SourmashSignature]]:
        """
        Perform parallel genome sketching from a FASTA file.

        Args:
            fasta_file (str): Path to the FASTA file.
            cores (int, optional): Number of parallel cores. Defaults to 1.
            ksize (int, optional): K-mer size. Defaults to 51.
            scale (int, optional): Scaling factor for MinHash. Defaults to 10_000.
            assigned_genome_name (str, optional): Name for the genome signature. Defaults to "full_genome".
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Tuple[sourmash.SourmashSignature, Dict[str, sourmash.SourmashSignature]]:
                The full genome signature and a dictionary of chromosome signatures.
        """
        self.logger.info("Starting parallel genome sketching with %d cores.", cores)
        fa_reader = SequenceReader(fasta_file, comment=True)
        mh_full = sourmash.MinHash(n=0, ksize=ksize, scaled=scale)
        chr_to_mh: Dict[str, sourmash.MinHash] = {}

        mh_lock = threading.Lock()
        chr_lock = threading.Lock()

        def process_sequence(
            name: str, seq: str, comment: Optional[str]
        ) -> None:
            header = f"{name} {comment}" if comment else name
            seq_type, seq_name = self.parse_fasta_header(header)
            current_mh = sourmash.MinHash(n=0, ksize=ksize, scaled=scale, track_abundance=True)
            current_mh.add_sequence(seq, force=True)

            with mh_lock:
                mh_full.merge(current_mh)

            if seq_type in {"sex", "autosome", "mitochondrial"}:
                with chr_lock:
                    key = f"{seq_type}-{seq_name}"
                    if seq_type == "mitochondrial":
                        key = "mitochondrial-M"
                    if key not in chr_to_mh:
                        chr_to_mh[key] = current_mh
                    else:
                        chr_to_mh[key].merge(current_mh)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
                futures = [
                    executor.submit(
                        process_sequence,
                        name,
                        seq,
                        comment
                    )
                    for name, seq, comment in fa_reader
                ]

                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error("Error processing sequence: %s", e)

        except Exception as e:
            self.logger.error("Error during parallel genome sketching: %s", e)
            raise

        mh_full_signature = sourmash.SourmashSignature(mh_full, name=assigned_genome_name)
        chr_signatures = {
            name: sourmash.SourmashSignature(mh, name=name)
            for name, mh in chr_to_mh.items()
        }

        self.logger.info("Parallel genome sketching completed.")
        return mh_full_signature, chr_signatures

    def amplicon_sketching(
        self,
        fasta_file: str,
        ksize: int = 51,
        scale: int = 10_000,
        amplicon_name: str = "amplicon",
    ) -> sourmash.SourmashSignature:
        """
        Create a FracMinHash sketch for an amplicon.

        Args:
            fasta_file (str): Path to the FASTA file.
            ksize (int, optional): K-mer size. Defaults to 51.
            scale (int, optional): Scaling factor for MinHash. Defaults to 10_000.
            amplicon_name (str, optional): Name of the amplicon. Defaults to "amplicon".

        Returns:
            sourmash.SourmashSignature: The resulting amplicon signature.
        """
        self.logger.info("Starting amplicon sketching for: %s", amplicon_name)
        try:
            fa_reader = SequenceReader(fasta_file)
            mh_full = sourmash.MinHash(n=0, ksize=ksize, scaled=scale)
            for _, seq in fa_reader:
                mh_full.add_sequence(seq, force=True)

            amplicon_sig = sourmash.SourmashSignature(mh_full, name=amplicon_name)
            self.logger.info("Amplicon sketching completed for: %s", amplicon_name)
            return amplicon_sig

        except Exception as e:
            self.logger.error("Error during amplicon sketching: %s", e)
            raise

    # *******************************
    # *        Exporting            *
    # *******************************

    @staticmethod
    def export_sigs_to_zip(
        sigs: List[sourmash.SourmashSignature], output_file: str
    ) -> None:
        """
        Export a list of signatures to a ZIP file.

        Args:
            sigs (List[sourmash.SourmashSignature]): List of Sourmash signatures.
            output_file (str): Path to the output ZIP file.

        Raises:
            ValueError: If the output file does not have a .zip extension.
            FileExistsError: If the output file already exists.
        """
        if not output_file.lower().endswith(".zip"):
            raise ValueError("Output file must have a .zip extension.")

        if os.path.exists(output_file): 
            raise FileExistsError("Output file already exists.")

        try:
            with sourmash.save_load.SaveSignatures_ZipFile(output_file) as save_sigs:
                for signature in sigs:
                    save_sigs.add(signature)
        except Exception as e:
            logging.error("Failed to export signatures to zip: %s", e)
            raise

