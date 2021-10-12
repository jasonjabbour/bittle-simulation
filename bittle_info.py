import xml.etree.ElementTree as ET

bittle = ET.parse('models/bittle_modified.urdf')
root = bittle.getroot()

total_mass = 0
for mass in root.iter('mass'):
    total_mass += float(mass.attrib['value'])
    print(mass.attrib)

print(f'Bittle total mass:{round(total_mass,4)} kilograms')

# for child in root:
#     print(child.tag, child.attrib)

