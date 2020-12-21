for i in range(1,17):
        print ("if opcion == %s:"%(i))
        print ("    df%s = pd.DataFrame()" %(i))
        print ("    df%s['Fecha'] = pd.to_datetime(f%s)" % (i,i))
        print ("    df%s.index = df%s['Fecha']" % (i,i))
        print ("    df%s['Casos'] = c%s" % (i,i))
        print ("    n=df%s" %(i))
        print("")

