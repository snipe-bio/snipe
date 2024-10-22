class SigType:
    SAMPLE = "SAMPLE"
    GENOME = "GENOME"
    AMPLICON = "AMPLICON"

    def __eq__(self, other):
        if isinstance(other, SigType):
            return (self.SAMPLE == other.SAMPLE and
                    self.GENOME == other.GENOME and
                    self.AMPLICON == other.AMPLICON)
        elif isinstance(other, str):
            return other in (self.SAMPLE, self.GENOME, self.AMPLICON)
        return False

    def __repr__(self):
        return f"SigType.{self.SAMPLE}"

    def __str__(self):
        return self.SAMPLE