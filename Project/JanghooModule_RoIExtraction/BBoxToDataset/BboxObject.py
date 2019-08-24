class BboxObject :
    def __init__(self, p1, p2) :
        x1 = p1[0]
        y1 = p1[1]
        x2 = p2[0]
        y2 = p2[1]
        if x1 > x2 :
            self.xmax = int(x1)
            self.xmin = int(x2)
        else :
            self.xmax = int(x2)
            self.xmin = int(x1)

        if y1 > y2 :
            self.ymax = int(y1)
            self.ymin = int(y2)
        else :
            self.ymax = int(y2)
            self.ymin = int(y1)

        # Debug
        # print(self.xmin, self.ymin, "~", self.xmax, self.ymax)