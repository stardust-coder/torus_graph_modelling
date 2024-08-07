# filenames and suffix from EEG data(Chennu et al., 2016)

def get_eeg_filenames():
    ind_list = {
        2: ["000","003", "006", "014"],
        3: ["003", "008", "021", "026"],
        5: ["004", "009", "022", "027"],
        6: ["003", "008", "013", "026"],
        7: ["003", "008", "021", "027"],
        8: ["004", "010", "015", "028"],
        9: ["003", "008", "021", "026"],
        10: ["005", "010", "015", "028"],
        13: ["003", "008", "013", "026"],
        14: ["007","011", "016", "031"],
        18: ["003", "009", "014", "027"],
        20: ["004", "009", "022", "027"],
        22: ["004", "009", "014","015"],
        23: ["003", "008", "022", "027"],
        24: ["003", "010", "015", "028"],
        25: ["003", "008", "021", "026"],
        26: ["003", "008", "013", "026"],
        27: ["001", "010", "023", "028"],
        28: ["004", "011", "016", "029"],
        29: ["005", "010", "023", "028"],
    }
    file_name_list = {
        2: "02-2010-anest 20100210 135.",
        3: "03-2010-anest 20100211 142.",
        5: "05-2010-anest 20100223 095.",
        6: "06-2010-anest 20100224 093.",
        7: "07-2010-anest 20100226 133.",
        8: "08-2010-anest 20100301 095.",
        9: "09-2010-anest 20100301 135.",
        10: "10-2010-anest 20100305 130.",
        13: "13-2010-anest 20100322 132.",
        14: "14-2010-anest 20100324 132.",
        18: "18-2010-anest 20100331 140.",
        20: "20-2010-anest 20100414 131.",
        22: "22-2010-anest 20100415 132.",
        23: "23-2010-anest 20100420 094.",
        24: "24-2010-anest 20100420 134.",
        25: "25-2010-anest 20100422 133.",
        26: "26-2010-anest 20100507 132.",
        27: "27-2010-anest 20100823 104.",
        28: "28-2010-anest 20100824 092.",
        29: "29-2010-anest 20100921 142.",
    }
    return ind_list, file_name_list
