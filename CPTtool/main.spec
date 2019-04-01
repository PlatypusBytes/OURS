# -*- mode: python -*-
a = Analysis(['cpt_tool.py'],
             pathex=['.'],
             hiddenimports=['numpy.core._dtype_ctypes'],
             hookspath=None,
             runtime_hooks=None)
			 
for d in a.datas:
    if 'pyconfig' in d[0]: 
        a.datas.remove(d)
        break
				
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas + [(r".\shapefiles\Robertson.dbf", r".\shapefiles\Robertson.dbf", "DATA"),
                     (r".\shapefiles\Robertson.shp", r".\shapefiles\Robertson.shp", "DATA")
                     ],
          name='CPTtool.exe',
          debug=False,
          strip=False,
          upx=True,
          console=True,
		  icon='.\\delIcon.ico',
		  version='.\\version.txt')
