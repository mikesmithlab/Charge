from labvision.video import ReadVideo
import matplotlib.pyplot as plt


if __name__ == '__main__':

    colour = [[0,0]]
    for frame in ReadVideo('W:/GranularCharge/TwoTone/2023_07_05/P1001789.mp4'):
        colour.append(np.sum(np.sum(frame[:,:,0])),np.sum(np.sum(frame[:,:,2])))

