from fastsam import FastSAM, FastSAMPrompt

model = FastSAM('./weights/FastSAM-x.pt')
IMAGE_PATH = './images/dogs.jpg'
DEVICE = 'cpu'
everything_results = model(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

# everything prompt
#ann = prompt_process.everything_prompt()
ann = prompt_process.box_prompt(bboxes=[[570,200,800,600],[560,190,810,610]])
#ann = prompt_process.rbox_prompt(rboxes=[[[570,200],[800,600],[800,200],[570,600]]])
#ann = prompt_process.point_prompt(points=[[620, 360]], pointlabel=[1])

prompt_process.plot(annotations=ann,output_path='./output/dogrbox.jpg',)
