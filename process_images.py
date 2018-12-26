import glob
from PIL import Image

img_paths = glob.glob('original_images/*/*.jpg')


def rotate_and_crop(img, horizontal_position_start):
    img_rotate_5 = img.rotate(5)
    img_rotate_10 = img.rotate(10)
    img_rotate_15 = img.rotate(15)
    img_rotate_minus_5 = img.rotate(-5)
    img_rotate_minus_10 = img.rotate(-10)
    img_rotate_minus_15 = img.rotate(-15)

    horizontal_position_end = horizontal_position_start + 425

    img_org = img.crop((horizontal_position_start, 156, horizontal_position_end, 581))
    img_rotate_5 = img_rotate_5.crop((horizontal_position_start, 156, horizontal_position_end, 581))
    img_rotate_10 = img_rotate_10.crop((horizontal_position_start, 156, horizontal_position_end, 581))
    img_rotate_15 = img_rotate_15.crop((horizontal_position_start, 156, horizontal_position_end, 581))
    img_rotate_minus_5 = img_rotate_minus_5.crop((horizontal_position_start, 156, horizontal_position_end, 581))
    img_rotate_minus_10 = img_rotate_minus_10.crop((horizontal_position_start, 156, horizontal_position_end, 581))
    img_rotate_minus_15 = img_rotate_minus_15.crop((horizontal_position_start, 156, horizontal_position_end, 581))

    img_org.save('images/{0}-{1}-center.jpg'.format(img.filename[16:][:-3], horizontal_position_start))
    img_rotate_5.save('images/{0}-{1}-rotate_5.jpg'.format(img.filename[16:][:-3], horizontal_position_start))
    img_rotate_10.save('images/{0}-{1}-rotate_15.jpg'.format(img.filename[16:][:-3], horizontal_position_start))
    img_rotate_15.save('images/{0}-{1}-rotate_15.jpg'.format(img.filename[16:][:-3], horizontal_position_start))
    img_rotate_minus_5.save('images/{0}-{1}-rotate_minus_5.jpg'.format(img.filename[16:][:-3], horizontal_position_start))
    img_rotate_minus_10.save('images/{0}-{1}-rotate_minus_10.jpg'.format(img.filename[16:][:-3], horizontal_position_start))
    img_rotate_minus_15.save('images/{0}-{1}-rotate_minus_15.jpg'.format(img.filename[16:][:-3], horizontal_position_start))


for img in img_paths:
    img = Image.open(img)

    rotate_and_crop(img, 350)
    rotate_and_crop(img, 360)
    rotate_and_crop(img, 370)
    rotate_and_crop(img, 380)
    rotate_and_crop(img, 390)
    rotate_and_crop(img, 400)
    rotate_and_crop(img, 410)
    rotate_and_crop(img, 420)
    rotate_and_crop(img, 430)
    rotate_and_crop(img, 440)
    rotate_and_crop(img, 450)
    rotate_and_crop(img, 460)
    rotate_and_crop(img, 470)
    rotate_and_crop(img, 480)
    rotate_and_crop(img, 490)
    rotate_and_crop(img, 500)
    rotate_and_crop(img, 510)
    rotate_and_crop(img, 520)
