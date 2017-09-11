from xml.etree.ElementTree import parse
from os import walk,listdir,makedirs,remove
from os.path import splitext,split,exists
from os import path
import natsort

def mkdir(bpath, dname=None):
    """

    create absolute path in string type
    if path is not exist, make directory first then create path.
    :param bpath:  root path
    :param dname: directory name
    :return: string
    """
    if dname is None:
        _path = bpath
    else:
        _path = bpath+'/'+dname

    if not exists(_path ):
        makedirs(_path )
    return str(_path )


def rm(dpath):
    flist = listdir(dpath)
    for file in flist:
        remove(dpath+'/'+file)

def read_text(path,fname):
    lines = []
    with open(path+'/'+fname+'.txt', 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            lines.append(line.strip())

    return lines


def filelist(path, ext= 0):
    files =[]
    files_ext =[]

    raw = listdir(path)
    for file in raw:
        files.append(splitext(file)[0])
        files_ext.append(file)

    if ext:
        return natsort.natsorted(files_ext)
    else:
        return natsort.natsorted(files)


def get_file_list(rpath):
    video_name = []
    video_ext = []
    gt_ext = []
    raw = listdir(rpath)
    for file in raw:
        if splitext(file)[-1] == '.xml':
            gt_ext.append(file)
        else:
            video_ext.append(file)
            video_name.append(splitext(file)[0])
    #assert(len(video_ext) == len(gt_ext)) ,'every video should have ground truth'
    return video_name,video_ext,gt_ext


def isExist(bpath, fname,pr=1):
    return path.exists(bpath + '/' + fname)