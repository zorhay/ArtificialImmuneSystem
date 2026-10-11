from PIL import Image, ImageChops, ImageOps

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def resize_image(img, heigth, output_path='ktrac.jpg'):
    w, h = img.size
    width = int(heigth * w / h)
    img.thumbnail((28, 28))
    # img.resize((width, heigth))
    img.save(output_path, 'JPEG')
    return img


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('--output', default='ktrac.jpg')
    args = parser.parse_args()
    image_path = args.image
    image = Image.open(image_path).convert('L')
    image.show()
    image = trim(image)
    # image = resize_image(image, 28)
    # image = ImageOps.fit(image, (28, 28), Image.Resampling.LANCZOS)
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    image.save(args.output, 'JPEG')
    image.show()


    # im = Image.open('img_4.jpg').convert('L')
    # avatar_size = (59, 59)
    # method = Image.NEAREST if im.size == avatar_size else Image.Resampling.LANCZOS
    # formatted_im = ImageOps.fit(im, avatar_size, method=method, centering=(0.5, 0.5))
    # formatted_im.save('foo.jpg', 'JPEG', quality=95)