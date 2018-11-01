# coding=utf-8
"""
Some useful tools to process .xml files
"""

def removePrefix(root):
    """
    remove all tag's prefix in element tree
    :param root: xml element tree
    :return: void
    """
    if '}' in root.tag:
        root.tag = root.tag.split('}')[1]
    if not root:
        return
    for child in root:
        removePrefix(child)



def findSubelement(root, str):
    """
    find the first subelement whose key is 'str', searching by dfs algorithm.
    :param root: xml element tree
    :param str: key of object element
    :return: object element
    """
    if not root.text:
        return None
    #print("Element: ", root.tag, " ", root)
    if root.tag == str:
        return root
    for child in root:
        r = findSubelement(child, str)
        if r != None and r.tag == str: return r
    return None