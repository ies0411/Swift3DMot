import uuid


class LabelInformationBuilder:
    def __init__(self, label_interface):
        self.label_interface = label_interface
        self.objects_dict = {}

    def __len__(self):
        return len(self.objects_dict)

    def build_object_id(self):
        return str(uuid.uuid4())

    def get_class_info(self, class_name):
        object_classes = self.label_interface['object_tracking']['object_classes']
        for object_class in object_classes:
            if object_class['name'] == class_name:
                return object_class
        raise ValueError(f'unknown class name {class_name}')

    def get_property_info(self, class_name, property_name):
        class_info = self.get_class_info(class_name)
        for property_info in class_info['properties']:
            if property_info['name'] == property_name:
                return property_info
        raise ValueError(f'unknown property name {property_name}')

    def get_option_id(self, option_name, property_info):
        for opt in property_info['options']:
            if opt['name'] == option_name:
                return opt['id']
        raise ValueError(f'unknown option name {option_name}')

    def build_object_frame_info(self, class_name, frame_num, position, rotation, size):
        class_info = self.get_class_info(class_name)
        frame_info = {
            "num": frame_num,
            "properties": [],  # should be filled when you need to define per_frame property
            "annotation": {
                "coord": {
                    "position": position,
                    "rotation_quaternion": rotation,
                    "size": size,
                },
                "meta": {"visible": True, "alpha": 1, "color": class_info['color']},
            },
        }
        return frame_info

    # for example, if you want to add "Motion" property with "Walking" option,
    # you need to give input property_dict as { "Motion": "Walking" } (radio type), { "Is carrying": ["Child", "Baggage"]} (checkbox type)
    def build_object_properties(self, class_name, property_dict):
        result_properties = []
        for property_name, option_value in property_dict.items():
            property_info = self.get_property_info(class_name, property_name)
            if property_info['type'] == 'radio':
                option_id = self.get_option_id(option_value, property_info)
                result_properties.append(
                    {
                        'type': 'radio',
                        'property_id': property_info['id'],
                        'property_name': property_name,
                        'option_id': option_id,
                        'option_name': option_value,
                    }
                )
            elif property_info['type'] == 'checkbox':
                if not isinstance(option_value, list):
                    raise ValueError('option value should be list for checkbox type')
                valid_option_names = []
                valid_option_ids = []
                for option_name in option_value:
                    option_id = self.get_option_id(option_name, property_info)
                    valid_option_names.append(option_name)
                    valid_option_ids.append(option_id)
                result_properties.append(
                    {
                        'type': 'checkbox',
                        'property_id': property_info['id'],
                        'property_name': property_name,
                        'option_id': valid_option_ids,
                        'option_name': valid_option_names,
                    }
                )
            elif property_info['type'] == 'free response':
                if not isinstance(option_value, str):
                    raise ValueError(
                        'option value should be str for free response type'
                    )
                result_properties.append(
                    {
                        'type': 'free response',
                        'property_id': property_info['id'],
                        'property_name': property_name,
                        'value': option_value,
                    }
                )
            else:
                continue
        return result_properties

    def create_object(self, object_id, class_name, frame_info):
        if object_id in self.objects_dict:
            # raise ValueError('object id already exist')
            return
        else:
            max_tracking_id = 0
            for k, v in self.objects_dict.items():
                if v['tracking_id'] > max_tracking_id:
                    max_tracking_id = v['tracking_id']
            max_tracking_id += 1

            class_info = self.get_class_info(class_name)
            self.objects_dict[object_id] = {
                "id": object_id,
                "class_id": class_info['id'],
                "tracking_id": max_tracking_id,
                "class_name": class_name,
                "annotation_type": "cuboid",
                "frames": [frame_info],
                "properties": [],
            }

    def append_object(self, object_id, frame_info):
        if object_id in self.objects_dict:
            self.objects_dict[object_id]['frames'].append(frame_info)
        else:
            # raise ValueError('object with given object id does not exist')
            return

    def create_or_append_object(
        self,
        object_id,
        class_name,
        frame_num,
        position,
        rotation,
        size,
        property_dict={},
        per_frame_property_dict={},
    ):
        frame_info = self.build_object_frame_info(
            class_name, frame_num, position, rotation, size
        )
        if object_id in self.objects_dict:
            self.append_object(object_id, frame_info)
            self.objects_dict[object_id]['frames'][-1][
                'properties'
            ] = self.build_object_properties(class_name, per_frame_property_dict)
        else:
            self.create_object(object_id, class_name, frame_info)
            self.objects_dict[object_id]['properties'] = self.build_object_properties(
                class_name, property_dict
            )

    def build_label_information_json(self):
        objects = []
        for object_id, object_info in self.objects_dict.items():
            objects.append(object_info)

        label_info_to_upload = {
            "version": "0.6.3",
            "meta": {
                "image_info": {},
                "edit_info": {
                    "brightness": 0,
                    "contrast": 0,
                    "elapsed_time": 0,
                    "objects": [],
                    "canvas_scale": 1,
                    "timeline_scale": 1,
                },
            },
            "result": {"objects": [], "categories": {"frames": [], "properties": []}},
            "tags": {
                "classes_id": [],
                "categories_id": [],
                "class": [],
                "classes_count": [],
                "time_spent": 0,
            },
        }
        label_info_to_upload["result"]["objects"] = objects

        classes_id = set()
        classes = set()
        classes_count = {}
        for obj in label_info_to_upload["result"]["objects"]:
            classes_id.add(obj["class_id"])
            classes.add(obj["class_name"])
            if obj['class_name'] in classes_count:
                classes_count[obj["class_name"]] += 1
            else:
                classes_count[obj["class_name"]] = 1
        label_info_to_upload["tags"]["classes_id"] = list(classes_id)
        label_info_to_upload["tags"]["class"] = list(classes)
        label_info_to_upload["tags"]["classes_count"] = [
            {"id": self.get_class_info(k)['id'], "name": k, "count": v}
            for k, v in classes_count.items()
        ]

        return label_info_to_upload
