class CUSUM: # cumulative sum is the mechanism with which we implement change detection
    def __init__(self, M, eps, h):
        self.M = M  # M is going to be used to compute the reference point
        self.eps = eps
        self.h = h
        self.t = 0
        self.reference = 0
        self.g_plus = 0
        self.g_minus = 0

    def update(self, sample):
        self.t += 1
        if self.t < self.M:  # the first M samples are used to 
            self.reference += sample / self.M
            return 0
        else:
            s_plus = (sample - self.reference) - self.eps
            s_minus = -(sample - self.reference) - self.eps
            self.g_plus = max(0, self.g_plus + s_plus)
            self.g_minus = max(0, self.g_minus + s_minus)
            return self.g_plus > self.h or self.g_minus > self.h

    def reset(self):  # after a change detection we must reset all the parameters, and we do this by calling this function
        self.t = 0
        self.g_minus = 0
        self.g_plus = 0
        self.reference = 0
