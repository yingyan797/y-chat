# from utils.language import question_entailment

class Processor:
    def __init__(self, text:str, seglen=400):
        self.text = text
        self.sgl = seglen
        self._segmentation()

    def _segmentation(self, rwd=0.12):
        self.segments = []
        # if self.sgl >= len(self.text):
        #     self.segments.append((0, len(self.text), False))
        #     return [self.text]
        i = 0
        while i < len(self.text):
            ei = i + self.sgl
            while ei < len(self.text) and self.text[ei].isalnum():
                ei += 1
            self.segments.append([i, ei, False])
            i = ei-int(self.sgl*rwd)
            while i < len(self.text) and self.text[i].isalnum():
                i += 1
        print(self.segments)
    
    def entailment(self, q, thr=0.9):
        for seg in self.segments:
            score = question_entailment(q, self.text[seg[0]:seg[1]])
            if score and score[0]["score"] > thr:
                seg[2] = True
        if len(self.segments) < 2:
            return self.segments
        i = 0
        j = -2
        while i < len(self.segments):
            if self.segments[i][2]:
                if i == j+1:
                    seg = self.segments.pop(i)
                    self.segments[j][1] = seg[1]
                else:
                    j = i
                    i += 1
            else:
                if i+1 < len(self.segments):
                    self.segments[i][1] = self.segments[i+1][0]
                if i > 0 and self.segments[i-1][2]:
                    self.segments[i][0] = self.segments[i-1][1]
                j = -2
                i += 1
        return self.segments
        


