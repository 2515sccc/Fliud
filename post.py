import itasca as it 

for i in range(13):
    name = str(i+1)
    name1 = 'vel'+str(i+1)
    name2 = 'dis'+str(i+1)
    it.command("""
    res {}
    plot bitmap plot 1 filename '{}' size 2560 1440
    plot bitmap plot 2 filename '{}' size 2560 1440
    """.format(name,name1,name2))