import os
import re
import datetime


def read_revision():
    release = '1'
    os.system('svnversion > tmpVers.txt')
    revisionFile = open('tmpVers.txt', 'r')
    revision = (revisionFile.readline()).split(':')[-1]
    version = revision.strip()
    version = re.findall('\d+', version)[0]
    revisionFile.close()
    os.remove('tmpVers.txt')

    # date
    now = datetime.datetime.now()

    with open('version.txt', 'w') as f:
        f.write("VSVersionInfo(\n" +
                "ffi=FixedFileInfo(\n" +
                "filevers=(" + str(release) + ", 0, 0, " + str(version) + "),\n" +
                "prodvers=(" + str(release) + ", 0, 0, " + str(version) + "),\n" +
                "mask=0x3f,\n" +
                "flags=0x0,\n" +
                "OS=0x40004,\n" +
                "fileType=0x1,\n" +
                "subtype=0x0,\n" +
                "date=(0, 0)\n" +
                "),\n" +
                "kids=[\n" +
                "StringFileInfo(\n" +
                "[\n" +
                "StringTable(\n" +
                "u'040904B0',\n" +
                "[StringStruct(u'CompanyName', u'Deltares'),\n" +
                "StringStruct(u'FileDescription', u'CPTtool'),\n" +
                # "StringStruct(u'FileVersion', u'" + str(version) + "'),\n" +
                "StringStruct(u'InternalName', u'CPTtool'),\n" +
                "StringStruct(u'LegalCopyright', u'Deltares" + r" \xae " + str(now.year) + "'),\n" +
                "StringStruct(u'OriginalFilename', u'CPTtool.exe'),\n" +
                "StringStruct(u'ProductName', u'CPTtool" + r" \xae " + "Deltares'),\n" +
                "StringStruct(u'ProductVersion', u'" + str(version) + "')])\n" +
                "]), \n" +
                "VarFileInfo([VarStruct(u'Translation', [1033, 1200])])\n" +
                "]\n" +
                ")")


def compile_code():
    os.system(r"RMDIR /S /Q .\dist")
    os.system(r"RMDIR /S /Q .\build")
    os.system(r"..\Python37\Scripts\pyinstaller.exe --onefile main.spec")
    os.remove(r"./version.txt")


if __name__ == '__main__':
    read_revision()
    compile_code()
