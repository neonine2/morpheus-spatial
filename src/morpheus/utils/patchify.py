import numpy as np
import pandas as pd

def verify_input_data(df, required_columns):
    if not set(required_columns).issubset(df.columns):
        raise ValueError(
            f"Dataframe does not contain all required columns: {required_columns}"
        )
    return True

def image_to_patch(df, 
                   patch_dim, 
                   width=None, 
                   height=None, 
                   recinterval=True):
    
    df = df.copy(deep=True)
    df['PatchNumber'], df['x0'], df['y0'] = np.nan, np.nan, np.nan
    df['PatchNumber'] = df['PatchNumber'].astype('Int64')
    pth_x = patch_dim[0]
    pth_y = patch_dim[1]
    if width is None and height is None:
        width = int(np.ceil(np.max(df.Location_Center_X)/patch_dim[0]))
        height = int(np.ceil(np.max(df.Location_Center_Y)/patch_dim[1]))
    for ii in range(width):
        for jj in range(height):
            x0, x1 = ii*pth_x,(ii+1)*pth_x
            y0, y1 = jj*pth_y,(jj+1)*pth_y
            in_Patch = df['Location_Center_X'].between(x0,x1,inclusive="left") & \
                                df['Location_Center_Y'].between(y0,y1,inclusive="left")
            if np.sum(in_Patch)>0:
                df.loc[in_Patch,'PatchNumber'] = ii*height+jj
                if recinterval:
                    df.loc[in_Patch,'x0'] = x0
                    df.loc[in_Patch,'y0'] = y0
    return df

def get_patch_coord(df, patch_sz):
    df = df.copy(deep=True)
    df = df[['ImageNumber','PatchNumber', 'x0','y0']]
    coord = df.groupby(['ImageNumber','PatchNumber']).mean()[['x0','y0']].groupby('ImageNumber').mean().reset_index(drop=True).to_numpy()/patch_sz
    return coord.astype('int')

def patch_to_pixel(df, width, height, pixel_dim):
    df = df.copy(deep=True)
    df.loc[:,'Location_Center_X'] = df['Location_Center_X']-df['x0']
    df.loc[:,'Location_Center_Y'] = df['Location_Center_Y']-df['y0']
    
    # assign new ImageNumber such that each patch is now considered a unique image
    val = (df['ImageNumber']-1)*(df['PatchNumber'].max()+1) + df['PatchNumber']
    df['original_ImageNumber'] = df['ImageNumber']
    df['ImageNumber'] = val.astype(int)
    df = image_to_patch(df,pixel_dim,width,height,recinterval=False)
    return df

def patch_to_matrix(df, width, height, typeName, celltype, genelist, channel_to_remove=[]):
    df = df.copy(deep=True)
    if not isinstance(celltype,list):
        celltype = [celltype]
    for cell in celltype:
        df[cell] = df[typeName]==cell
    genes_to_keep = [gene for gene in genelist if gene not in set(channel_to_remove)]
    df = df[['ImageNumber','PatchNumber','original_ImageNumber']+genes_to_keep+celltype]
    nchannel = len(genes_to_keep)
    label = df.groupby(['ImageNumber']).max()[celltype]
    
    groupedpixel = df.groupby(['ImageNumber','PatchNumber'])
    groupedsum = groupedpixel.sum()
    groupedsum['original_ImageNumber'] = groupedpixel.mean()['original_ImageNumber']
    groupedsum = groupedsum.reset_index()
    groupedimage = groupedsum.groupby(['ImageNumber'])
    list_of_image = [v for k, v in groupedimage]
    nsample = len(list_of_image)

    intensity = np.zeros([nsample, nchannel, height, width])
    for i, image in enumerate(list_of_image):
        linear_index = image['PatchNumber'].tolist()
        col, row = np.unravel_index(linear_index, (height, width))
        row = width - row - 1
        intensity[i,:,row,col] = image[genes_to_keep].to_numpy()
    return intensity, label, genes_to_keep, groupedsum


