import xml.etree.ElementTree as ET
from pyvis.network import Network
import os


def create_knowledge_graph(xml_path, image_name, output_folder="knowledge_graphs"):
    """
    Parses the annotations XML file, creates a knowledge graph for the specified image,
    and saves it as an interactive HTML file.

    Args:
        xml_path (str): The path to the annotations XML file.
        image_name (str): The filename of the image to create the graph for.
        output_folder (str): The folder to save the generated HTML file in.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_html_file = os.path.join(output_folder, f"knowledge_graph_{os.path.splitext(image_name)[0]}.html")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # --- Graph Styling Configuration ---
    OBJECT_NODE_COLOR = "#007bff"  # Blue for main objects
    ATTRIBUTE_NODE_COLOR = "#ffc107"  # Yellow for attributes
    RELATIONSHIP_EDGE_COLOR = "#cccccc"  # Light grey for relationships
    ATTRIBUTE_EDGE_COLOR = "#6c757d"  # Darker grey for attribute links

    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=True,
                  cdn_resources='in_line')

    target_image = root.find(f".//image[@name='{image_name}']")

    if target_image is None:
        print(f"Image '{image_name}' not found in the XML file.")
        return None

    # --- 1. Add Object and Attribute Nodes ---
    for mask in target_image.findall('mask'):
        mask_id = mask.get('id')
        mask_label = mask.get('label')

        # Add the main object node
        net.add_node(mask_id, label=mask_label, color=OBJECT_NODE_COLOR, shape="dot", size=25)

        # --- Add attribute nodes ---
        for attr in mask.findall('attribute'):
            attr_name = attr.get('name')
            attr_value = attr.text

            # Create a unique ID for the attribute node to avoid collisions
            attr_id = f"{mask_id}_{attr_name}"
            attr_label = f"{attr_name}: {attr_value}"

            # Add the attribute node with a different shape and color
            net.add_node(attr_id, label=attr_label, color=ATTRIBUTE_NODE_COLOR, shape="square", size=15)

            # Add an edge from the object to its attribute
            net.add_edge(mask_id, attr_id, color=ATTRIBUTE_EDGE_COLOR, dashes=True)

    # --- 2. Add Relationship Edges ---
    for mask in target_image.findall('mask'):
        mask_id = mask.get('id')
        for rel in mask.findall('relationship'):
            target_id = rel.get('with')
            rel_label = rel.text

            # Add the main relationship edge with a distinct color
            net.add_edge(mask_id, target_id, label=rel_label, color=RELATIONSHIP_EDGE_COLOR)

    net.show_buttons(filter_=['physics', 'nodes', 'edges'])

    # Manually write the HTML file with UTF-8 encoding
    try:
        html_content = net.generate_html()
        with open(output_html_file, "w", encoding="utf-8") as f:
            f.write(html_content)
    except Exception as e:
        print(f"An error occurred while writing the HTML file: {e}")
        return None

    return output_html_file