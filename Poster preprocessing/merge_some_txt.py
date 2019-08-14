import os
def merge_txt(readDir, writeDir):
    outfile=open(writeDir,"r+")
    lines_seen = set()
    for line in outfile:
        lines_seen.add(line)
    print(lines_seen)
    f = open(readDir,"r")
    for line in f:
        if line not in lines_seen:
            outfile.write(line)
            lines_seen.add(line)
    outfile.close()
    print("merge success")
    f.close()
    os.remove(readDir)
    print("remove success")

merge_txt("data/Adventure.txt", "data/Action.txt")
merge_txt("data/Documentary.txt", "data/Biography.txt")
merge_txt("data/Thriller.txt", "data/Horror.txt")
merge_txt("data/Musical.txt", "data/Music.txt")

