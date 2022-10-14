# Dynamic_Contrast

import matplotlib.pyplot as plt
import numpy as np
import cv2
import dc_prctice as dcp

def rgb2ycbcr(img):
    row, col, channel = img.shape
    b = img[..., 0]
    g = img[..., 1]
    r = img[..., 2]
    y = np.zeros((row, col), dtype=np.float32)
    cr = np.zeros((row, col), dtype=np.float32)
    cb = np.zeros((row, col), dtype=np.float32)
    for i in range(row):
        for j in range(col):
            y[i][j] = 0.299 * r[i][j] + 0.587 * g[i][j] + 0.114 * b[i][j]
            cr[i][j] = (r[i][j] - y[i][j]) * 0.713 + 128
            cb[i][j] = (b[i][j] - y[i][j]) * 0.564 + 128

    return y.astype(np.uint8), cr.astype(np.uint8), cb.astype(np.uint8)

def ycbcr2rgb(y, cr, cb):
    mergedImg = cv2.merge([y, cr, cb])
    newImg = cv2.cvtColor(mergedImg, cv2.COLOR_YCrCb2BGR)
    return newImg

def findMean(arr):
    size = len(arr)
    mean = 0
    for i in range(size):
        mean += arr[i]
    mean = mean / size
    return mean

def findMedian(arr):
    size = len(arr)
    if size % 2:
        middle = int(size / 2) + 1
        median = arr[middle]
    else:
        midRight = int(size / 2)
        midLeft = midRight - 1
        median = (arr[midRight] + arr[midLeft]) / 2
    return median

def findMinMax(minMax, arr):
    '''
    :param minMax: false for min, true for max
    :return: min or max
    '''
    if minMax:
        return arr[0]
    else:
        return arr[-1]

def centPixConst(img, row, col):
    '''
    Using n * n matrix block, in this case n = 3
    [x, x, x]
    [x, x, x]
    [x, x, x]
    Thus there are 4 directional vectors
    '''
    #Return array containg directional local contrasts
    vecL = []

    n, directions = 3, 4

    dirVec = [[(-1, 0), (1, 0)],
              [(0, -1), (0, 1)],
              [(1, -1), (-1, 1)],
              [(-1, -1), (1, 1)]]

    coef = 1 / (n-1)

    for k in range(directions):
        c_k = 0
        for m in range(2):
            xdir, ydir = dirVec[k][m]
            # Directional Cordinate
            xdir += col
            ydir += row
            # Get directional vector value to check out of bound
            centVecVal = int(img[row][col])
            dirVecVal = int(img[ydir][xdir])
            if centVecVal and dirVecVal:
                numerator = centVecVal - dirVecVal
                denominator = centVecVal + dirVecVal
                localConst = abs(numerator / denominator)
                c_k += localConst
        c_k *= coef
        vecL.append(c_k)
    vecL.sort()
    return vecL

def locConstVal(arr):
    compare = []
    for i in range(len(arr)):
        compare.append(int(arr[i]*10))
    pureCompare = [*set(compare)]
    pairCnt = len(compare) - len(pureCompare)
    if pairCnt == 2:
        # Min edge with more darker val
        if pureCompare[0] == compare[0]:
            return findMinMax(False, arr)
        # Max edge with more brighter val
        elif pureCompare[0] == compare[-1]:
            return findMinMax(True, arr)
    # Median when center is not similar to neighbor
    elif pairCnt == 1:
        return findMedian(arr)
    # Mean when neighbors are similar
    else:
        return findMean(arr)

def sigmoidTrans(intensity, imin, imax, alpha):
    mxSubmn = imax - imin
    degree = -(alpha) * (intensity - 0.5 * (mxSubmn))
    newVal = (mxSubmn) / (1 + np.exp(degree))
    return newVal + imin

def localAdaptiveCont(img, iter):
    # Determine coefficient alpha
    alpha = 0.027
    # Determine thresholds T_low and T_high
    t_low, t_high = 0, 0.06

    #Brightness level : differ by intensity
    minVal, maxVal = 0, 255
    # Dynamic Contrast
    rows, cols = img.shape
    newImg = np.copy(img)

    #Iterate to test out parameters
    for _ in range(iter):
        prevImg = np.copy(newImg)
        #Calculate APL
        totPix = 0
        totOg = 0
        totNew = 0

        #Iterate each pixel
        for row in range(rows-1):
            for col in range(cols-1):
                totPix += 1
                totOg += prevImg[row][col]
                # Estimate the contrast of the central pixel C_k
                vecL = centPixConst(prevImg, row, col)
                # Estimate the value of the local contrast C
                c = locConstVal(vecL)
                # If T_low < c < T_high
                if (t_low <= c) and (c <= t_high):
                    # Transform the image
                    newVal = sigmoidTrans(prevImg[row][col], minVal, maxVal, alpha)
                    newImg[row][col] = newVal
                    totNew += newImg[row][col]
                # Else
                else:
                    totNew += newImg[row][col]
                    # Don't transform
                    continue
        '''
        ogApl = totOg / totPix
        newApl = totNew / totPix
        difApl = ogApl - newApl

        for row in range(rows-1):
            for col in range(cols-1):
                updateVal = newImg[row][col]
                #if new image becomes darker
                if difApl > 0:
                    if updateVal < 200:
                        newImg[row][col] = updateVal + difApl
                #if new image becomes brighter
                else:
                    if updateVal < 50:
                        updateVal += difApl
                        if updateVal > 0:
                            newImg[row][col] = updateVal
        '''
    return newImg

def main():
    imgone = r"D:\dc_hk\sample1\1.jpg"
    imgtwo = r"D:\dc_hk\sample1\2.bmp"
    imgthree = r"D:\dc_hk\sample1\3.jpg"
    imgfour = r"D:\dc_hk\sample1\4.bmp"
    imgfive = r"D:\dc_hk\sample1\5.jpg"

    imgArr = [imgone, imgtwo, imgthree, imgfour, imgfive]
    #imgArr = [imgone]
    # Read image
    for i in range(len(imgArr)):
        img = plt.imread(imgArr[i])
        Y, Cr, Cb = rgb2ycbcr(img)

        histArr = dcp.genHist(Y)
        pdfArr = dcp.genPdf(histArr)
        cdfArr, output = dcp.genCdf(pdfArr)

        eqArr = dcp.histEqual(Y, output)
        eqHistArr = dcp.genHist(eqArr)
        eqPdfArr = dcp.genPdf(eqHistArr)
        eqCdfArr, _ = dcp.genCdf(eqPdfArr)

        enhancedY = localAdaptiveCont(Y, 1)
        enhancedHistArr = dcp.genHist(enhancedY)
        enhancedPdfArr = dcp.genPdf(enhancedHistArr)
        enhancedCdfArr, _ = dcp.genCdf(enhancedPdfArr)

        saveLoc = r"D:\dc_hk\img" + str(i)

        ogGraph = saveLoc + "og_graph.png"
        dcp.visualization(Y, pdfArr, cdfArr, ogGraph)
        eqGraph = saveLoc + "eq_graph.png"
        dcp.visualization(eqArr, eqPdfArr, eqCdfArr, eqGraph)
        dcGraph = saveLoc + "dc_graph.png"
        dcp.visualization(enhancedY, enhancedPdfArr, enhancedCdfArr, dcGraph)

        ogImg = ycbcr2rgb(Y, Cb, Cr)
        ogName = saveLoc + "og.png"
        cv2.imwrite(ogName, ogImg)

        eqImg = ycbcr2rgb(eqArr, Cb, Cr)
        eqName = saveLoc + "eq.png"
        cv2.imwrite(eqName, eqImg)

        dcImg = ycbcr2rgb(enhancedY, Cb, Cr)
        dcName = saveLoc+ "dc.png"
        cv2.imwrite(dcName, dcImg)



if __name__ == "__main__":
    main()




import matplotlib.pyplot as plt
import numpy as np

def rgb2ycbcr(img):
    row, col, channel = img.shape
    b = img[..., 0]
    g = img[..., 1]
    r = img[..., 2]
    y = np.zeros((row, col), dtype=np.float32)
    cr = np.zeros((row, col), dtype=np.float32)
    cb = np.zeros((row, col), dtype=np.float32)
    for i in range(row):
        for j in range(col):
            y[i][j] = 0.299 * r[i][j] + 0.587 * g[i][j] + 0.114 * b[i][j]
            cr[i][j] = (r[i][j] - y[i][j]) * 0.713 + 128
            cb[i][j] = (b[i][j] - y[i][j]) * 0.564 + 128

    return y.astype(np.uint8), cr.astype(np.uint8), cb.astype(np.uint8)

def genHist(arr):
    row, col = arr.shape
    histArr = np.zeros((256, ), dtype=int)
    for i in range(row):
        for j in range(col):
            histArr[arr[i][j]] += 1
    return histArr

def genPdf(arr):
    sumArr = sum(arr)
    pdfArr = []
    for i in arr:
        prob = i / sumArr
        pdfArr.append(prob)
    return pdfArr

def genCdf(arr):
    probSum = 0
    cdfArr = []
    #2^8=256
    mapArr = [0 for _ in range(257)]
    for i in range(len(arr)):
        probSum += arr[i]
        cdfArr.append(probSum)
        mapArr[i] = int(probSum * 255)
    return cdfArr, mapArr

def histEqual(arr, mapArr):
    row, col = arr.shape
    #eqArr = np.zeros((row, col), dtype=int)
    eqArr = np.copy(arr)
    for i in range(row):
        for j in range(col):
            eqArr[i][j] = mapArr[arr[i][j]+1]
    return eqArr

def visualization(img, pdfArr, cdfArr, savePath):
    fig, (ogImg, pdf) = plt.subplots(2)
    fig.suptitle('Dynamic Contrast Test')

    ogImg.imshow(img, cmap='gray')

    #pdf.plot(pdfArr, color='black', alpha=0.5)
    pdf.bar(range(0, 256), pdfArr, color='black', alpha=0.5)
    pdf.set_ylabel('PDF', color='black', rotation=90)

    cdf = pdf.twinx()
    cdf.plot(cdfArr, color='red', alpha=0.5)
    cdf.set_ylabel('CDF', color='red', rotation=0)

    plt.savefig(savePath)
    #plt.show()
    return

def main():
    imgFile = r"D:\dc_hk\sample1\5.jpg"
    #imgFile = r"D:\dc_hk\lotte.jpg"

    img = plt.imread(imgFile)
    '''
    print(type(img))
    print(img.ndim)
    print(img.shape)
    '''
    Y, Cr, Cb = rgb2ycbcr(img)

    histArr = genHist(Y)
    pdfArr = genPdf(histArr)
    cdfArr, output = genCdf(pdfArr)

    eqArr = histEqual(Y, output)
    eqHistArr = genHist(eqArr)
    eqPdfArr = genPdf(eqHistArr)
    eqCdfArr, _ = genCdf(eqPdfArr)

    visualization(Y, pdfArr, cdfArr)

    visualization(eqArr, eqPdfArr, eqCdfArr)


if __name__ == "__main__":
    main()



