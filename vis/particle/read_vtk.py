def vtk (filename):
    
    # based on code by: Guillaume Jacquenot
    #  - https://stackoverflow.com/questions/11727822/reading-a-vtk-file-with-python
    # Adapted for reading Athena 4.2 files by Patryk Pjanka, 2021, using athena_read.py from Athena++ github repository.
    
    result = {}

    # map Athena4.2 variable names to Athena++ format
    in2out = {'density':'rho', 'velocity':'vel', 'pressure':'press'}
    
    # get time from the filename and out_dt
    time = int(filename.split('.')[-2]) * out_dt
    result['Time'] = time

    import numpy as np
    from vtk import vtkStructuredPointsReader, vtkInformation
    from vtk.util import numpy_support as VN

    reader = vtkStructuredPointsReader()
    reader.SetFileName(filename)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()

    # read in the file
    data = reader.GetOutput()

    # process coordinates
    x = np.zeros(data.GetNumberOfPoints())
    y = np.zeros(data.GetNumberOfPoints())
    z = np.zeros(data.GetNumberOfPoints())
    for i in range(data.GetNumberOfPoints()):
        x[i],y[i],z[i] = data.GetPoint(i)
    x = np.sort(np.unique(np.array(x).flatten()))
    y = np.sort(np.unique(np.array(y).flatten()))
    z = np.sort(np.unique(np.array(z).flatten()))
    result['x1f'] = x
    result['x2f'] = y
    result['x3f'] = z
    result['x1v'] = 0.5*(x[:-1]+x[1:])
    result['x2v'] = 0.5*(y[:-1]+y[1:])
    result['x3v'] = 0.5*(z[:-1]+z[1:])

    # process scalars
    dim = [i-1 if i>1 else i for i in data.GetDimensions()]
    for quantity in ['density','pressure']:
        u = VN.vtk_to_numpy(data.GetCellData().GetArray(quantity))
        u = u.reshape(dim,order='F')
        # squeeze removes dimensions of 1 if we're 2D
        result[in2out[quantity]] = np.squeeze(np.array(1.*u)).transpose()
        del u

    # process vectors
    dim = [i-1 if i>1 else i for i in data.GetDimensions()] + [3,]
    for quantity in ['velocity',]:
        u = VN.vtk_to_numpy(data.GetCellData().GetArray(quantity))
        u = u.reshape(dim,order='F')
        u = np.moveaxis(u, -1, 0)
        result[in2out[quantity]+'1'] = np.squeeze(np.array(1.*u[0])).transpose()
        result[in2out[quantity]+'2'] = np.squeeze(np.array(1.*u[1])).transpose()
        result[in2out[quantity]+'3'] = np.squeeze(np.array(1.*u[2])).transpose()
        del u
        
    return result