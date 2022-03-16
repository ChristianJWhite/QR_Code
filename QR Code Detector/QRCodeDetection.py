
from matplotlib import pyplot
from matplotlib.patches import Polygon, Rectangle

import imageIO.png
import math
import PIL


def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)

# This method packs together three individual pixel arrays for r, g and b values into a single array that is fit for
# use in matplotlib's imshow method
def prepareRGBImageForImshowFromIndividualArrays(r,g,b,w,h):
    rgbImage = []
    for y in range(h):
        row = []
        for x in range(w):
            triple = []
            triple.append(r[y][x])
            triple.append(g[y][x])
            triple.append(b[y][x])
            row.append(triple)
        rgbImage.append(row)
    return rgbImage
    

# This method takes a greyscale pixel array and writes it into a png file
def writeGreyscalePixelArraytoPNG(output_filename, pixel_array, image_width, image_height):
    # now write the pixel array as a greyscale png
    file = open(output_filename, 'wb')  # binary mode is important
    writer = imageIO.png.Writer(image_width, image_height, greyscale=True)
    writer.write(file, pixel_array)
    file.close()



#This method computes a greyscale representation from the red, green and blue pixel arrays 
def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    calculation = 0
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    
    for i in range(image_height):
        for z in range(image_width):
            calculation += (pixel_array_r[i][z] * 0.299)
            
            calculation += (pixel_array_g[i][z] * 0.587)
            
            calculation += (pixel_array_b[i][z] * 0.114)

            calculation = round(calculation)
            greyscale_pixel_array[i][z]=calculation
            calculation = 0
    
    return greyscale_pixel_array



#This method computes a contrast stretching from the minimum and maximum values of the input pixel array to the full 8 bit range of values between 0 and 255
def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    
    minMaxTuple = computeMinAndMaxValues(pixel_array, image_width, image_height)
    if (minMaxTuple[1] - minMaxTuple[0]) == 0:
        calculation = 0
    else:
        calculation = 255/(minMaxTuple[1] - minMaxTuple[0])
    
    for i in range(image_height):
        for z in range(image_width):
            pixel_array[i][z] = round((pixel_array[i][z] - minMaxTuple[0]) * calculation)
    
    return pixel_array



#This method returns a tuple (min_value, max_value) containing the smallest and largest value of a pixel array image
def computeMinAndMaxValues(pixel_array, image_width, image_height):
    minMax = [255,0]
    
    for i in range(image_height):
        for z in range(image_width):
            if pixel_array[i][z] < minMax[0]:
                minMax[0] = pixel_array[i][z]
            
            if pixel_array[i][z] > minMax[1]:
                minMax[1] = pixel_array[i][z]

    minMaxTuple = (minMax[0],minMax[1])
    return minMaxTuple



#This method computes and returns an image of the vertical edges using a 3x3 Sobel kernel
def computeVerticalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    edge_array = [[0.0]*image_width for _ in range(image_height)]
    calculation = 0.0
    
    for i in range(1,image_height-1):
        for j in range(1,image_width-1):
            calculation = ((0.0 - pixel_array[i-1][j-1] - (2 * pixel_array[i][j-1]) - pixel_array[i+1][j-1] +
                        pixel_array[i-1][j+1] + (2 * pixel_array[i][j+1]) + pixel_array[i+1][j+1]) / 8.0)
            edge_array[i][j] = calculation
                
    return edge_array



#This method computes and returns an image of the horizontal edges using a 3x3 Sobel kernel
def computeHorizontalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    edge_array = [[0.0]*image_width for _ in range(image_height)]
    calculation = 0.0
    
    for i in range(1,image_height-1):
        for j in range(1,image_width-1):
            calculation = ((0.0 + pixel_array[i-1][j-1] + (2 * pixel_array[i-1][j]) + pixel_array[i-1][j+1] -
                        pixel_array[i+1][j-1] - (2 * pixel_array[i+1][j]) - pixel_array[i+1][j+1]) / 8.0)
            edge_array[i][j] = calculation
                
    return edge_array



#This method computes the magnitude
def computeEdgeMagnitude(pixel_array, v_array, h_array, image_width, image_height):
    calculation = 0.0
    for i in range(image_height):
        for j in range(image_width):
            calculation = math.sqrt(((v_array[i][j])*(v_array[i][j])) + ((h_array[i][j])*(h_array[i][j])))
            pixel_array[i][j] = calculation
    return pixel_array



#This method contains a 3x3 mean filter
def computeBoxAveraging3x3(pixel_array, image_width, image_height):
    box_average = [[0.0]*image_width for _ in range(image_height)]
    calculation = 0.0
    
    for i in range(1,image_height-1):
        for j in range(1,image_width-1):
            calculation = (0.0 + pixel_array[i-1][j-1] +  pixel_array[i-1][j] + pixel_array[i-1][j+1] +
                        pixel_array[i][j-1] +  pixel_array[i][j] + pixel_array[i][j+1] +
                        pixel_array[i+1][j-1] + pixel_array[i+1][j] + pixel_array[i+1][j+1]) / 9.0
            box_average[i][j] = calculation
                
    return box_average



#This method performs a threshholding operation
def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] < threshold_value:
                pixel_array[i][j] = 0
            else:
                pixel_array[i][j] = 255
    return pixel_array



#This method performs a dilation
def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    bordered_array = createInitializedGreyscalePixelArray(image_width +2, image_height +2 )
    
    for i in range(image_height):
        for j in range(image_width):
            bordered_array[i+1][j+1] = pixel_array[i][j] 
    
    for p in range(image_height):
        for q in range(image_width):
            if (bordered_array[p][q] == bordered_array[p+1][q+1])and(
                bordered_array[p][q+1] == bordered_array[p+1][q+1])and(
                bordered_array[p][q+2] == bordered_array[p+1][q+1])and(
                bordered_array[p+1][q] == bordered_array[p+1][q+1])and(
                bordered_array[p+1][q+2] == bordered_array[p+1][q+1])and(
                bordered_array[p+2][q] == bordered_array[p+1][q+1])and(
                bordered_array[p+2][q+1] == bordered_array[p+1][q+1])and(
                bordered_array[p+2][q+2] == bordered_array[p+1][q+1])and(
                bordered_array[p+1][q+1]==0):
                pixel_array[p][q] = 0
            else:
                pixel_array[p][q] = 1

    return pixel_array



#This method performs an erosion
def computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    bordered_array = createInitializedGreyscalePixelArray(image_width +2, image_height +2 )
    
    for i in range(image_height):
        for j in range(image_width):
            bordered_array[i+1][j+1] = pixel_array[i][j] 
    
    for p in range(image_height):
        for q in range(image_width):
            if (bordered_array[p][q] == bordered_array[p+1][q+1])and(
                bordered_array[p][q+1] == bordered_array[p+1][q+1])and(
                bordered_array[p][q+2] == bordered_array[p+1][q+1])and(
                bordered_array[p+1][q] == bordered_array[p+1][q+1])and(
                bordered_array[p+1][q+2] == bordered_array[p+1][q+1])and(
                bordered_array[p+2][q] == bordered_array[p+1][q+1])and(
                bordered_array[p+2][q+1] == bordered_array[p+1][q+1])and(
                bordered_array[p+2][q+2] == bordered_array[p+1][q+1])and(
                bordered_array[p+1][q+1]!=0):
                pixel_array[p][q] = 1
            else:
                pixel_array[p][q] = 0

    return pixel_array



#Queue class
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)



#This method performs a connected component analysis
def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    new_array = createInitializedGreyscalePixelArray(image_width, image_height)
    padded_array = createInitializedGreyscalePixelArray(image_width+2, image_height+2)
    current_label = 1
    q = Queue()
    sz_dict = {}
    
    #adds zero padding border
    for i in range(image_height):
        for j in range(image_width):
            padded_array[i+1][j+1] = pixel_array[i][j] 
    
    for y in range(image_height):
        for x in range(image_width):
            
            #padded_array == 1
            if (padded_array[y+1][x+1]):
                location = [y+1, x+1]
                q.enqueue(location)

                while q.isEmpty() == False:
                    location[0] = (q.items[q.size()-1])[0]
                    location[1] = (q.items[q.size()-1])[1]

                    #iterates "current_label" keys into dictionary
                    if (current_label) in sz_dict:
                        sz_dict[current_label] += 1
                    else:
                        sz_dict[current_label] = 1
                            
                    new_array[location[0]-1][location[1]-1] = current_label
                    padded_array[location[0]][location[1]] = 0
                    
                    #calculates if pixel is object and not yet visited
                    if ((padded_array[(location[0])][(location[1])-1])):
                            q.enqueue([(location[0]),(location[1])-1])
                            padded_array[location[0]][location[1]-1] = 0
                                    
                                    
                    if ((padded_array[(location[0])][(location[1])+1])):
                            q.enqueue([(location[0]),(location[1])+1])
                            padded_array[location[0]][location[1]+1] = 0
                    
                    
                    if ((padded_array[(location[0])-1][(location[1])])):
                            q.enqueue([(location[0])-1,(location[1])])
                            padded_array[location[0]-1][location[1]] = 0
                               
                                    
                    if ((padded_array[(location[0])+1][(location[1])])):
                            q.enqueue([(location[0])+1,(location[1])])
                            padded_array[location[0]+1][location[1]] = 0
                    
                    q.dequeue()
                current_label += 1
            else:
                pass

    return(new_array,sz_dict)



# This method finds the MINIMUM x and y coordinates of foreground pixels
def minPix(pixel_array, image_width, image_height):
    pixel = []
    iHeight = image_height
    jWidth = image_width
    for i in range(image_height):
        for j in range(image_width):
            if(pixel_array[i][j] > 0):
                if (j < jWidth):
                    jWidth = j
    for i in range(image_height):
        for j in range(image_width):
            if((pixel_array[i][j] > 0)and(pixel_array[i][jWidth+4] > 0)):
                if (i < iHeight):
                    iHeight = i
    pixel = [jWidth,iHeight]
    return pixel



#This method finds the bottom right pixel of the QR code (maximum both x and y values)
def bottomRightPix(pixel_array, image_width, image_height):
    pixel = []
    iHeight = 0
    jWidth = 0
    for i in range(image_height):
        for j in range(image_width):
            if(pixel_array[i][j] > 0):
                if (j > jWidth):
                    jWidth = j
    for i in range(image_height):
        for j in range(image_width):
            if((pixel_array[i][j] > 0)and(pixel_array[i][jWidth-4] > 0)):
                if (i > iHeight):
                    iHeight = i
    pixel = [jWidth,iHeight] 
    return pixel


#This method find the top right pixel of the QR code (maximum x and minimum y values)
def topRightPix(pixel_array, image_width, image_height):
    pixel = []
    jWidth = 0
    iHeight = image_height
    for i in range(image_height):
        for j in range(image_width):
            if(pixel_array[i][j] > 0):
                if (i <= iHeight):
                    iHeight = i
    for i in range(image_height):
        for j in range(image_width):
            if((pixel_array[i][j] > 0)and(pixel_array[iHeight+4][j])):
                if (j > jWidth):
                    jWidth = j
    pixel = [jWidth,iHeight]
    return pixel


#This method find the bottom left pixel of the QR code (minimum x and maximum y values)
def bottomLeftPix(pixel_array, image_width, image_height):
    pixel = []
    jWidth = image_width
    iHeight = 0
    for i in range(image_height):
        for j in range(image_width):
            if((pixel_array[i][j] > 0)and(i > iHeight)):      
                iHeight = i
    for i in range(image_height):
        for j in range(image_width):
            if((pixel_array[i][j] > 0)and(j < jWidth)and(pixel_array[iHeight-4][j] > 0)):
                jWidth = j
    pixel = [jWidth,iHeight]
    return pixel

##############################################################################################################
###################################### CHANGE FILENAME BELOW #################################################
##############################################################################################################


def main():
    filename = "./images/shanghai.png"
    smoothed_array = []
    closed_array = []
    tl = []
    threshold_value = 70

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(filename)

    #converts rgb values to greyscale
    pixel_array = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)

    #stretches between 0 and 255
    stretched_pixel_array = scaleTo0And255AndQuantize(pixel_array, image_width, image_height)

    #vertical edge
    vertical_array = computeVerticalEdgesSobelAbsolute(stretched_pixel_array, image_width, image_height)

    #horizontal edge
    horizontal_array = computeHorizontalEdgesSobelAbsolute(stretched_pixel_array, image_width, image_height)

    #edge magnitude
    edge_array = computeEdgeMagnitude(stretched_pixel_array, vertical_array, horizontal_array, image_width, image_height)

    #smooth over the edge magnitude and stretches
    smoothed_array = edge_array
    for i in range(8):
        smoothed_array = computeBoxAveraging3x3(smoothed_array, image_width, image_height)
    smoothed_array = scaleTo0And255AndQuantize(smoothed_array, image_width, image_height)

    #thresholding operation
    binary_array = computeThresholdGE(smoothed_array, threshold_value, image_width, image_height)

    #performs a morphological closing operation to fill holes
    closed_array = binary_array
    for i in range(5):
        closed_array = computeDilation8Nbh3x3FlatSE(closed_array, image_width, image_height)
        closed_array = computeErosion8Nbh3x3FlatSE(closed_array, image_width, image_height)

    #connected component analysis
    (ccimg,ccsizes) = computeConnectedComponentLabeling(closed_array, image_width, image_height)
    keymax = max(ccsizes, key=ccsizes.get)
    
    for i in range(image_height):
        for j in range(image_width):
            if(ccimg[i][j]!= keymax):
                ccimg[i][j] = 0

    #top left position of bounding box
    tl = minPix(ccimg, image_width, image_height)
    
    #bottom right position of bounding box
    br = bottomRightPix(ccimg, image_width, image_height)

    #top right position of the bounding box
    tr = topRightPix(ccimg, image_width, image_height)

    #bottom left position of the bounding box
    bl = bottomLeftPix(ccimg, image_width, image_height)

    ################################################
    #creates image
    pyplot.imshow(PIL.Image.open(filename))
    #pyplot.imshow(ccimg, cmap="gray")

    # get access to the current pyplot figure
    axes = pyplot.gca()

    # creates a polygon shaped bounding box in clockwise order
    #TL = top left, TR = top right, BR = bottom right, BL = bottom left
    poly = Polygon([tl,tr,br,bl], linewidth=3, edgecolor='g', facecolor='none')
    
    # paint the polygon rectangle over the current plot
    axes.add_patch(poly)   
    #axes.add_patch(rect)


    # plot the current figure
    pyplot.show()

    

if __name__ == "__main__":
    main()
