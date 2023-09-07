from labvision.video import ReadVideo, WriteVideo
from filehandling import BatchProcess


if __name__ == '__main__':
    path = 'Y:/GranularCharge/ChargeProject/DelrinGlassBeaker_8kV_60sinterval/'
    filefilter = path + '00?.mp4'
    print(filefilter)

    writevid = WriteVideo(path + 'joined.mp4',frame_size=(1024, 1280, 3))

    for file in [path + '001.mp4', path + '002.mp4', path + '003.mp4']:
        readvid = ReadVideo(file)

        for i in range(readvid.num_frames):
            img = readvid.read_next_frame()
            writevid.add_frame(img)
        readvid.close()
    writevid.close()