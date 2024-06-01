import numpy as np
import mne
from mne.viz import ClickableImage, plot_alignment, set_3d_view, snapshot_brain_montage

id = "003"
# load human eeg data series from PlosComp journal
FILE_NAME= f"02-2010-anest 20100210 135.{id}"
PATH_TO_DATA_DIR = "../../data/Sedation-RestingState/"
PATH_TO_DATA = PATH_TO_DATA_DIR + FILE_NAME + ".set"

def load_human_eeg(input_fname, events=None):
    data = mne.io.read_epochs_eeglab(
            input_fname, verbose=False, montage_units="cm")
    return data

raw = load_human_eeg(PATH_TO_DATA)

#montage = raw.get_montage()
#fig = plot_alignment(raw.info)
#set_3d_view(figure=fig, azimuth=20, elevation=80)
#xy,im = snapshot_brain_montage(fig,montage)


xy_pts = np.vstack([xy[ch] for ch in raw.ch_names])
xy_pts[:,1] = -xy[:,1] #just for plot issue

def draw_2d(est_dict,d):
    plt.figure(figsize=(10,10))
    for i,j in itertools.combinations(range(1,d+1),2):
        i -= 1
        j -= 1
        plt.plot([xy_pts[:,0][i].item(), xy_pts[:,0][j].item()], [xy_pts[:,1][i].item(), xy_pts[:,1][j].item()], color="green")

    for (i,j,k) in zip(xy_pts[:,0],xy_pts[:,1],range(1,d+1)):
        plt.annotate(k, xy=(i+1, j+1))
        
    plt.scatter(xy_pts[:,0],xy_pts[:,1], zorder=2)
    plt.savefig("visualize.png")


#montage = raw.get_montage()
#fig = plot_alignment(raw.info)
#set_3d_view(figure=fig, azimuth=20, elevation=80)
#xy,im = snapshot_brain_montage(fig,montage)

#fig2,ax = plt.subplot(figsize=(10,10))
#ax.imshow(im)
#ax.set_axis_off()
#fig2.savefig("./brain.png",bbox_inches="tight")
