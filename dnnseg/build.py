import sys
import os
import shutil
import pandas as pd
import argparse
#import paramiko


argparser = argparse.ArgumentParser('''
Compiles data for processing with DNN-Seg.
''')

argparser.add_argument('meta', help='Path to metadata repository. (Lists of files - e.g. files.txt and VAD information - e.g. vad.txt')
argparser.add_argument('wrd', help='Path to word information (e.g. data.wrd)')
argparser.add_argument('-d', '--de2', type=str, default=None, help='Path to de2 Speech Corpus')
argparser.add_argument('-s', '--sp1', type=str, default=None, help='Path to sp1 Speech Corpus')
argparser.add_argument('-o', '--outdir', default='./', help='Path to output directory (if not specified, uses current working directory)')
#argparser.add_argument('-b', '--bender', default=True, help='State whether output directory is remote server "bender" (True, default) or not')
args = argparser.parse_args()

# if args.bender:
    # ssh = paramiko.SSHClient()
    # ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # ssh.connect("129.70.104.163", port=22, username="kiara", password="bender123")
    # sftp = ssh.open_sftp()

german_files = []
spanish_files = []

with open(args.meta + "/german_files.txt", "r") as f:
    for l in f.readlines():
        german_files.append(l.strip()[:-4])
#print(german_files[:5])
    
with open(args.meta + "/spanish_files.txt", "r") as f:
    for l in f.readlines():
        spanish_files.append(l.strip()[:-4])


german_vad = pd.read_csv(args.meta + "/german_vad.txt", sep=' ', header=None, names=['fileID', 'start', 'end'])
german_vad.start = german_vad.start.round(3)
german_vad.end = german_vad.end.round(3)

german_wrd = pd.read_csv(args.wrd + '/german.wrd', sep=' ', header=None, names=['fileID', 'start', 'end', 'label'])
german_wrd.start = german_wrd.start.round(3)
german_wrd.end = german_wrd.end.round(3)


spanish_vad = pd.read_csv(args.meta + "/spanish_vad.txt", sep=' ', header=None, names=['fileID', 'start', 'end'])
spanish_vad.start = spanish_vad.start.round(3)
spanish_vad.end = spanish_vad.end.round(3)

# spanish_wrd = pd.read_csv(args.wrd + '/spanish.wrd', sep=' ', header=None, names=['fileID', 'start', 'end', 'label'])
# spanish_wrd.start = spanish_wrd.start.round(3)
# spanish_wrd.end = spanish_wrd.end.round(3)




sys.stderr.write('Processing German corpus data...\n')

if args.de2 is not None:
    if not os.path.exists(args.outdir + '/german'):
        os.makedirs(args.outdir + '/german')

    for fileID in german_files:
        subject = fileID[10:13]
        #print(subject)
        for i in reversed(range(1, 3)): ##subject folder could end in _01 or _02
            subject_folder = subject.replace("v", "VP ") + '_0' + str(i)
            #print(args.de2 + '/' + subject_folder)
            if os.path.exists(args.de2 + '/' + subject_folder):
                in_path = args.de2 + '/' + subject_folder + '/' + fileID + '.wav'
                out_path = args.outdir + '/german/' + fileID + '.wav'
                shutil.copy2(in_path, out_path)
                #print("in theory I copied something")

                to_print = german_vad[german_vad.fileID == fileID]
                to_print['speaker'] = subject
                to_print.to_csv(args.outdir + '/german/%s.vad' % fileID, sep=' ', index=False)

                to_print = german_wrd[german_wrd.fileID == fileID]
                to_print['speaker'] = subject
                to_print.to_csv(args.outdir + '/german/%s.vad' % fileID, sep=' ', index=False)

                break ##if 02 is found, search for 01 no longer necessary
else:
    sys.stderr.write('No path provided to German corpus. Skipping...\n')


sys.stderr.write('Processing Spanish corpus data...\n')
if args.sp1 is not None:
    if not os.path.exists(args.outdir + '/spanish'):
        os.makedirs(args.outdir + '/spanish')
    for fileID in spanish_files:
        subject = fileID[10:12]
        for i in reversed(range(1, 3)):  ##subject folder could end in _01 or _02
            subject_folder = subject.replace("v", "VP ") + '_0' + str(i)
            if os.path.exists(args.de2 + '/' + subject_folder):
                in_path = args.de2 + '/' + subject_folder + '/' + fileID + '.wav'
                out_path = args.outdir + '/spanish/' + fileID + '.wav'
                if args.bender:
                    sftp.put(in_path, out_path)
                else:
                    shutil.copy2(in_path, out_path)

                to_print = spanish_vad[spanish_vad.fileID == fileID]
                to_print['speaker'] = subject
                if args.bender:
                    with sftp.open('/spanish/%s.vad' % fileID , "w") as f:
                        f.write(to_print.to_csv(sep=' ', index=False))
                else:
                    to_print.to_csv(args.outdir + '/spanish/%s.vad' % fileID, sep=' ', index=False)

                to_print = spanish_wrd[spanish_wrd.fileID == fileID]
                to_print['speaker'] = subject
                if args.bender:
                    with sftp.open('/spanish/%s.wrd' % fileID , "w") as f:
                        f.write(to_print.to_csv(sep=' ', index=False))
                else:
                    to_print.to_csv(args.outdir + '/spanish/%s.wrd' % fileID, sep=' ', index=False)

                break  ##if 02 is found, search for 01 no longer necessary
else:
    sys.stderr.write('No path provided to Spanish data. Skipping...\n')
