import numpy as np
from PIL import Image

# 使用奇异值总和的百分比进行筛选
def svd(data,scale):
    # scale 代表你要保留的奇异值比例
    u,sigma,v = np.linalg.svd(data)
    svd_data = np.zeros(data.shape)
    total = sum(sigma)
    sum_data = 0
    for index,item in enumerate(sigma):  
        svd_data += item * np.dot(u[:,index].reshape(-1,1),v[index,:].reshape(1,-1))
        sum_data += item
        if sum_data >= scale * total:
            break
    return svd_data

def compress(data,scale):
    r = svd(data[:,:,0],scale)
    g = svd(data[:,:,1],scale)
    b = svd(data[:,:,2],scale)

    result = np.stack((r,g,b),2)
    result[result > 255] = 255
    result[result < 0] = 0
    result = result.astype(int)
    return result

if __name__ == '__main__':
    image = Image.open('test.jpg')
    width,height = image.size
    arr = np.zeros((width,height,3)) # RGB 
    for x in range(width):
        for y in range(height):
            arr[x][y] = image.getpixel((x, y))
    # 原生 range 不支持浮点数，所以用 np.arange 代替
    for c in np.arange(.1,.9,.2):
        result = compress(arr,c)
        for x in range(width):
            for y in range(height):
                image.putpixel((x, y),(result[x,y,0],result[x,y,1],result[x,y,2]))
        image.save('test_'+str(int(100 * c))+'%.jpg')

