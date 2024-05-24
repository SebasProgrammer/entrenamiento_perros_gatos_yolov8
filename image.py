from PIL import Image
from ultralytics import YOLO

model = YOLO('modelo2.pt')
results = model('perro.jpeg')  # results list
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('results.jpg')  # save image