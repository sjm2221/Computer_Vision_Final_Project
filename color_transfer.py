import numpy as np
import math
from skimage import io, color
import matplotlib.pyplot as plt

def ReshapeHistogram(Is, It, perc):
    #convert Is, It to CIELab color space
    #compute Smax
    #rgb = io.imread(filename)
    Is = color.rgb2lab(Is)
    It = color.rgb2lab(It)
    Io = Is.copy()

    V = 0.5
    I = np.concatenate([Is, It])
    B = math.ceil((np.max(I) - np.min(I))/V)
    Bmin = 10
    Smax = math.floor(math.log2(B/Bmin))
    print(Smax)
    I_min = np.min(I)
    
    for i in range(3):
        Is_c = Is[:,:,i]
        It_c = It[:,:,i]
        
        Hs = []
        Ht = []
        for j in range(B):
            hs = 0
            ht = 0
            for k in range(Is_c.shape[0]):
                for m in range(Is_c.shape[1]):
                    hs += P(Is_c[k, m], j, V, I_min)
            Hs.append((hs, I_min + (j)*V))

            for k in range(It_c.shape[0]):
                for m in range(It_c.shape[1]):
                    ht += P(It_c[k, m], j, V, I_min)
            Ht.append((ht, I_min + (j)*V))
        Hs_ = [h[0] for h in Hs]
        vs_ = [h[1] for h in Hs]
        Ht_ = [h[0] for h in Ht]
        vt_ = [h[1] for h in Ht]

        Hsk = np.array(Hs_)
        Htk = np.array(Ht_)
        for k in range(1, int(perc*Smax+1)):
            #Hsk = Hs_.copy()
            #Htk = Ht_.copy()
            #down / upsample Hs_
            for k_ in range(0, len(Hs), k):
                r_max = min(k_ + k, len(Hs))
                s = np.mean(Hs_[k_:r_max])
                Hsk[k_:r_max] = s
                t = np.mean(Ht_[k_:r_max])
                Htk[k_:r_max] = t

            Rmint = FindPeaks(Htk)
            Rmint = [r[0] for r in Rmint]
            for m in range(len(Rmint)-1):
                Hsk[Rmint[m]:Rmint[m+1]] = RegionTransfer(Hsk[Rmint[m]:Rmint[m+1]], Htk[Rmint[m]:Rmint[m+1]], k/Smax)
            
            Rmins = FindPeaks(Hsk)
            Rmins = [r[0] for r in Rmins]
            Hok = Hsk.copy()
            for m in range(len(Rmins)-1):
                Hok[Rmins[m]:Rmins[m+1]] = RegionTransfer(Hsk[Rmins[m]:Rmins[m+1]], Htk[Rmins[m]:Rmins[m+1]], k/Smax)
            Hsk = Hok
            #DrawHistogram(Hsk, I_min, V)
        Io[:,:,i] = HistMatch(Is_c, I_min, Hs_, Hsk, V)
    
    return color.lab2rgb(Io)

def P(i, j, V, min_I):
    if j == math.floor((i - min_I)/V+1):
        return 1
    return 0

def FindPeaks(H):
    H_ = np.gradient(H.copy())
    H_[H_ >= 0] = 1
    H_[H_ < 0] = -1
    H2 = np.gradient(H_)
    #print (H2)
    return np.argwhere(H2 < 0)

def RegionTransfer(Hs, Ht, wt):
    ws = 1-wt
    Ho = []
    for hs, ht in zip(Hs, Ht):
        Ho.append ((hs - ws*np.mean(Hs))*(wt*np.std(Ht)/(ws*np.std(Hs)) + wt*np.mean(Ht)))

    return np.array(Ho)
def HistMatch(Is, Imin, Hs, Ho, V=1):
    Cs = np.cumsum(Hs)
    Co = np.cumsum(Ho)

    Io = Is.copy()

    for i in range(Io.shape[0]):
        for j in range(Io.shape[1]):
            c = Cs[int((Is[i, j] - Imin)/V)]
            for k in range(len(Co)-1):
                if c >= Co[k] and c < Co[k+1]:
                    c = k
                    break
            Io[i, j] = Imin + (c)*V
    return Io

def DrawHistogram(H, Imin, V=1):
    plt.figure(1, figsize=(8, 6))
    plt.subplot(1,1,1)
    plt.xlabel('v')

    v = []

    for i in range(len(H)):
        v.append(Imin + (i)*V)

    plt.plot(v, H)
    plt.show()


def main():
    file1 = 'bricks_sq.jpg'
    file2 = 'pink_zigzag.jpg'
    I_s = io.imread(file1)
    I_t = io.imread(file2)
    I_o = ReshapeHistogram(I_s, I_t, perc=1.0)

    io.imsave('output.jpg', I_o)
    
    return

if __name__ == '__main__':
    main()