#!/usr/bin/python2.7

import rospy
#from std_msgs.msg import Header
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
import numpy as np
import cv2
import time
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import auto

#import sys
ns = ''
print("orbbec_frame")
print(ns)
file_path = '/home/har/yolov5_d435i/test_image1'

# prefix to the names of dummy fields we add to get byte alignment correct. this needs to not
# clash with any actual field names
DUMMY_FIELD_PREFIX = '__'

# mappings between PointField types and numpy types
type_mappings = [(PointField.INT8, np.dtype('int8')), (PointField.UINT8, np.dtype('uint8')), (PointField.INT16, np.dtype('int16')),
                 (PointField.UINT16, np.dtype('uint16')), (PointField.INT32, np.dtype('int32')), (PointField.UINT32, np.dtype('uint32')),
                 (PointField.FLOAT32, np.dtype('float32')), (PointField.FLOAT64, np.dtype('float64'))]

pftype_to_nptype = dict(type_mappings)
nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)

# sizes (in bytes) of PointField types
pftype_sizes = {PointField.INT8: 1, PointField.UINT8: 1, PointField.INT16: 2, PointField.UINT16: 2,
                PointField.INT32: 4, PointField.UINT32: 4, PointField.FLOAT32: 4, PointField.FLOAT64: 8}

print("program begin!")

# @converts_to_numpy(PointField, plural=True)
def fields_to_dtype(fields, point_step):
    '''Convert a list of PointFields to a numpy record datatype.
    '''
    offset = 0
    np_dtype_list = []
    for f in fields:
        while offset < f.offset:
            # might be extra padding between fields
            np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        dtype = pftype_to_nptype[f.datatype]
        if f.count != 1:
            dtype = np.dtype((dtype, f.count))

        np_dtype_list.append((f.name, dtype))
        offset += pftype_sizes[f.datatype] * f.count

    # might be extra padding between points
    while offset < point_step:
        np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
        offset += 1

    return np_dtype_list

def pointcloud2_to_array(cloud_msg, squeeze=True):
    ''' Converts a rospy PointCloud2 message to a numpy recordarrospkg'ay

    Reshapes the returned array to have shape (height, width), even if the height is 1.
    The reason for using np.frombuffer rather than struct.unpack is speed... especially
    for large point clouds, this will be <much> faster.
    '''
    # construct a numpy record type equivalent to the point type of this cloud
    dtype_list = fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)

    # parse the cloud into an array
    cloud_arr = np.frombuffer(cloud_msg.data, dtype_list)

    # remove the dummy fields that were added
    cloud_arr = cloud_arr[
        [fname for fname, _type in dtype_list if not (fname[:len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)]]

    if squeeze and cloud_msg.height == 1:
        return np.reshape(cloud_arr, (cloud_msg.width,))
    else:
        return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))

def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float):
    '''Pulls out x, y, and z columns from the cloud recordarray, and returns
	a 3xN matrix.
    '''
    # remove crap points
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]

    # pull out x, y, and z values
    points = np.zeros(cloud_array.shape + (3,), dtype=dtype)
    points[..., 0] = cloud_array['x']
    points[..., 1] = cloud_array['y']
    points[..., 2] = cloud_array['z']

    return points


def pointcloud2_to_xyz_array(cloud_msg, remove_nans=False):
    return get_xyz_points(pointcloud2_to_array(cloud_msg), remove_nans=remove_nans)


def read_and_save_data(cloud_msg, rgb_msg, bridge):
    t = time.time()
    try:
        rgb_image = bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        cloud = pointcloud2_to_xyz_array(cloud_msg)
        cloud[np.isnan(cloud)] = 0
        orbbec_rgb_name = '{}/{:.3f}.jpg'.format(file_path,t)
        cv2.imwrite(orbbec_rgb_name, rgb_image)

        img_depth = (cloud[..., 2]*1000).astype(np.uint16)
        
        orbbec_depth_name = '{}/{:.3f}.png'.format(file_path, t)
        cv2.imwrite(orbbec_depth_name, img_depth)
        
        # np.save('{}/{:.3f}.npy'.format(file_path, t), cloud)[:,0:3]

      
       
    
        
        #img_final = 2dimage(img_depth)

    except CvBridgeError as e:
        print(e)
    # print('Used time: {} ms'.format(1000 * (time.time() - t)))
    print(img_depth.shape)
    predict_result = auto.predict_frame(cloud[:,0:3])
    print(predict_result)
    print('orbbec frame saved')


def listener():
    rospy.init_node('image_recorder')

    cloud_msg = message_filters.Subscriber('/camera/depth/color/points', PointCloud2)
    bridge = CvBridge()
    rgb_msg = message_filters.Subscriber('/camera/color/image_raw', Image)

    fs = 3 #Hz
    rate = rospy.Rate(fs) # 3 Hz
    ts = message_filters.ApproximateTimeSynchronizer([cloud_msg, rgb_msg], fs, 10)
    ts.registerCallback(read_and_save_data, bridge)

    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass



