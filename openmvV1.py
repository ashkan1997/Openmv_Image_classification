import sensor, image, pyb, os, time, tf, uos, gc

TRIGGER_THRESHOLD = 2
BG_UPDATE_FRAMES = 50
BG_UPDATE_BLEND = 64

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time=2000)
sensor.set_auto_whitebal(False)
clock = time.clock()

extra_fb = sensor.alloc_extra_fb(sensor.width(), sensor.height(), sensor.RGB565)

print("About to save background image...")
sensor.skip_frames(time = 2000) # Give the user time to get ready.
extra_fb.replace(sensor.snapshot())
print("Saved background image - Now frame differencing!")

net = None
labels = None

try:
    net = tf.load("model.tflite", load_to_fb=uos.stat('model.tflite')[6] > (gc.mem_free() - (32*1024)))
except Exception as e:
    print(e)
    raise Exception('Failed to load "model.tflite" (' + str(e) + ')')

try:
    labels = [line.rstrip('\n') for line in open("labels.txt")]
except Exception as e:
    raise Exception('Failed to load "labels.txt" (' + str(e) + ')')
triggered = False
frame_count = 0

while True:
    clock.tick()
    img = sensor.snapshot()

    frame_count += 1
    if frame_count > BG_UPDATE_FRAMES:
        frame_count = 0
        img.blend(extra_fb, alpha=(256-BG_UPDATE_BLEND))
        extra_fb.replace(img)

    img.difference(extra_fb)

    hist = img.get_histogram()
    diff = hist.get_percentile(0.99).l_value() - hist.get_percentile(0.90).l_value()
    triggered = diff > TRIGGER_THRESHOLD

    if triggered:
        for obj in net.classify(img, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
            print("**********\nPredictions")
            img.draw_rectangle(obj.rect())
            predictions_list = list(zip(labels, obj.output()))

            for i in range(len(predictions_list)):
                print("%s = %f" % (predictions_list[i][0], predictions_list[i][1]))

    print(clock.fps(), triggered)
