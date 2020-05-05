from PIL import Image
import cv2

def add_mask(picture, width, height):
    # Process every pixel
    for x in range(0,width):
        for y in range(0,21):
            #current_color = picture.getpixel( (x,y) )
            new_color = (0, 255, 0)
            picture.putpixel( (x,y), new_color)

    for x in range(0,width):
        for y in range(190,height):
            new_color = (0, 255, 0)
            picture.putpixel( (x,y), new_color)

    for x in range(196, 240):
        a = int(85*x/44-357.6364)
        for y in range(21, a):
            new_color = (0, 255, 0) 
            picture.putpixel( (x,y), new_color)    
    return picture   

for i in range(2353):
    add1 = '/home/zwenbo/Documents/research/deform/rope_dataset'
    add2 = 'run05' # CHANGE
    if len(str(i)) == 1:
        add3 = '/img_000{}.jpg'.format(i)
    elif len(str(i)) == 2:
        add3 = '/img_00{}.jpg'.format(i)
    elif len(str(i)) == 3:
        add3 = '/img_0{}.jpg'.format(i) 
    elif len(str(i)) == 4:
        add3 = '/img_{}.jpg'.format(i) 
    picture = Image.open(add1 + '/rope/' + add2 + add3)
    width, height = picture.size
    picture = add_mask(picture, width, height)
    picture.save('./rope_dataset/rope_mask/run05' + add3)
# picture = Image.open("./rope_dataset/rope/run05/1.jpg")
# width, height = picture.size



#picture.save('./rope_dataset/rope/run05/2.jpg')
