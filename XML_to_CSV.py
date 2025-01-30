import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

xml_path = './Annotations'

def xml_to_csv(files):
    xml_list = []
    for xml_file in files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            bbx = member.find('bndbox')
            xmin = int(bbx.find('xmin').text)
            ymin = int(bbx.find('ymin').text)
            xmax = int(bbx.find('xmax').text)
            ymax = int(bbx.find('ymax').text)
            label = member.find('name').text
            value = (root.find('filename').text,
                    int(root.find('size')[0].text),
                    int(root.find('size')[1].text),
                    label,
                    xmin,
                    ymin,
                    xmax,
                    ymax
                    )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def get_xml_files(folder):
    xml_files = list()
    for file in os.listdir(folder):
        if file.endswith('.xml'):
            xml_files.append(os.path.join(folder, file))
    return xml_files

def main():
    xml_files = get_xml_files(xml_path)
    xml_df = xml_to_csv(xml_files)
    xml_df.to_csv(os.path.join(xml_path, 'labels.csv'), index=False)

if __name__ == '__main__':
    main()