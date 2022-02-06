import os
import sys

def create_labels_file(path_to_labels):
  label_map_str = """
  item {
      id: 1
      name: 'COTS'
      }
                  """
  if not os.path.exists(path_to_labels):
    os.mkdir(path_to_labels)

  if os.path.exists(path_to_labels+'/label_map.pbtxt') is False:
      with open(path_to_labels+'/label_map.pbtxt', 'w') as f:
          f.write(label_map_str)
      print('Successfully created label_map.pbtxt file')

if __name__=='__main__':
    LABELS_PATH=sys.argv[1]

    create_labels_file(LABELS_PATH)