# Author: Carlos Gozález <carlos-emiliano.gonzalez-gallardo@sorbonne-universite.fr>
# Copyright (C) 2020 Carlos González <carlos-emiliano.gonzalez-gallardo@sorbonne-universite.fr>
# Cite as: 
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import sys
import re

class Segmentation(object):
    """Segmentation class

    """

    def __init__(self, segmentation_file=None):
        """

        Parameters
        ----------
        segmentation_file : string, optional
            The location of the segmented file.
        """


        self.segmentation_file = segmentation_file

        """
        self.content : list of string
            List of tokens (words and </S>) with the content of the segmented file

        self.borders : list of int
            List of borders where 0 if word and 1 if </S>
        """
        self.content = list()
        self.borders = list()


    def load(self, segmentation_file=None):
        """Reads the segmented file into self.content. It normalizes by lowercasing, removing
            all punctuation marks and tranforming each new line into </S>.

        Parameters
        ----------
        segmentation_file : string, optional
            The location of the segmented file
        """

        if segmentation_file:
            self.segmentation_file = segmentation_file

        if not self.segmentation_file:
            print("No file to load.")
            sys.exit()

        try:
            with open(self.segmentation_file, 'r') as file_p:
                content = file_p.read().lower().strip()
        except FileNotFoundError as ex:
            print(ex)
            sys.exit()

        content = re.sub(r'[.:;!,?]', ' ', content)
        content = re.sub(r'\n', '</S> ', content)
        content = re.sub(r"[ ]{2,}", " ", content).strip()
        content += " </S>"

        self.content = content.split(" ")
        print("Segmentation {} loaded!".format(self.segmentation_file))

    def create_borders(self):
        """Creates a list of borders where 1 if word and 0 if </S>

        """

        for _ in self.content:
            if _ == "</S>":
                self.borders.pop()
                self.borders.append(1)
            else:
                self.borders.append(0)

        print("Segmentation {} has {} borders.".format(self.segmentation_file,
            self.borders.count(1)))

    def get_noseg(self):
        """Returns a list of words without the "</S>" identifier from the segmentation

        """
        return list(filter(lambda x: x != "</S>", self.content))


    def dump_borders(self):
        """Dumps borders into disk

        """

        dump_file = self.segmentation_file + ".seg"
        dump_content = "".join([str(_) for _ in self.borders])

        print("Dumping borders to file: {}".format(dump_file))
        with open(dump_file, 'w') as outfile:
            outfile.write(dump_content)
