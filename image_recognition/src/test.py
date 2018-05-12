

def limfocit_to_image(limfocit):
    img_array = []
    line = []
    for i in range(28):
        for j in range(28):
            if (i, j) in limfocit.white_pixels:
                line.append(True)
            else:
                line.append(False)
        img_array.append(line)
        line = []
    return img_array
                    