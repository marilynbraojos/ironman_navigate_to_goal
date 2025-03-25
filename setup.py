from setuptools import find_packages, setup

package_name = 'ironman_chase_object'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='burger',
    maintainer_email='marilynbraojos@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_detection_node=ironman_chase_object.object_detection_node:main',
            'view_image_raw=ironman_chase_object.view_image_raw:main',
            'get_object_range=ironman_chase_object.get_object_range:main',
            'velocity_controller=ironman_chase_object.velocity_controller:main',
        ],
    },
)
