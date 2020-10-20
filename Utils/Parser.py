import re
import Levenshtein as edit
from collections import OrderedDict
from pdfminer.pdfparser import PDFParser as mPDFParser, PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams


class PDFParser:
    def __init__(self):
        self.__resources_manager = PDFResourceManager()
        self.__params_manager = LAParams()
        self.__aggregator = PDFPageAggregator(rsrcmgr=self.__resources_manager, laparams=self.__params_manager)
        self.__interpreter = PDFPageInterpreter(rsrcmgr=self.__resources_manager, device=self.__aggregator)
        self.__analyzer = None

    def parse(self, input_path, password=""):
        result = []

        parser = mPDFParser(open(input_path, "rb"))
        document = PDFDocument()
        parser.set_document(document)
        document.set_parser(parser)
        document.initialize(password=password)

        for page in document.get_pages():
            self.__interpreter.process_page(page)
            layout = self.__aggregator.get_result()

            for out in layout:
                if hasattr(out, 'get_text'):
                    content = out.get_text().strip()
                    content = content.replace('\n', '|')
                    result.append((out.x0, out.height, content))

        return result

    def analyze(self, rule, parse_result):
        if rule == "Monetary Policy Report":
            self.__analyzer = MonetaryPolicyReportAnalyzer()
            result = self.__analyzer.analyze(parse_result)

        else:
            print("rule is invalid, doesn't analyze content")
            return parse_result

        return result


class MonetaryPolicyReportAnalyzer:
    def __init__(self):
        self.pages = OrderedDict()
        self.index_tree = IndexNode()
        self.most_height = 0
        self.most_x0 = 0
        self.title_max_length = 30
        self.pdf_name = ""

    def analyze(self, parse_result):
        self.pages = self.divide_to_pages(parse_result)
        self.pdf_name = self.get_pdf_name(self.pages)
        self.most_height = self.get_most_height(self.pages)
        self.pages = self.delete_non_text_part(self.pages)
        self.most_x0 = self.get_most_x0(self.pages)
        self.pages = self.merge_continuous_paragraph(self.pages)
        plain_text = self.concat_pages_to_plain_text(self.pages)
        self.index_tree = self.convert_to_tree(plain_text)

        return self.index_tree

    def get_most_height(self, pages_dict):
        contents = []
        for value in pages_dict.values():
            contents += value

        heights = [item[1] for item in contents]
        freq_dict = {}

        for height in heights:
            freq_dict[height] = freq_dict.setdefault(height, 0) + 1

        heights = [(k, v) for k, v in freq_dict.items()]
        heights.sort(key=lambda x: x[1])

        most_height = heights[-1][0]

        return most_height

    def get_most_x0(self, pages_dict):
        contents = []
        for value in pages_dict.values():
            contents += value

        x0s = [item[0] for item in contents]
        freq_dict = {}

        for x0 in x0s:
            freq_dict[x0] = freq_dict.setdefault(x0, 0) + 1

        x0s = [(k, v) for k, v in freq_dict.items()]
        x0s.sort(key=lambda x: x[1])

        most_x0 = x0s[-1][0]

        return most_x0

    def get_pdf_name(self, pages_dict):
        first_path_content = pages_dict['O']
        pdf_name = first_path_content[0][-1] + first_path_content[1][-1]

        return pdf_name

    def divide_to_pages(self, parse_result):
        cache = []
        in_page = False
        pages_dict = OrderedDict()
        parse_result = parse_result + [(0, 0, '')]

        for ind in range(len(parse_result)):
            line = parse_result[ind]
            content = line[-1]

            if content.strip() == "":
                if not in_page:
                    continue
                else:
                    page_index = cache[-1][-1].strip()
                    page_content = cache[:-1]

                    if not (page_index.isnumeric() or re.match('[IVX]+', page_index)):
                        # 首页
                        if not pages_dict:
                            page_index = 'O'
                            page_content = cache
                        else:
                            page_index = cache[0][-1].strip()
                            page_content = cache[1:]

                    pages_dict[page_index] = page_content
                    in_page = False
                    cache = []
            else:
                if not in_page:
                    in_page = True
                cache.append(line)

        return pages_dict

    def delete_non_text_part(self, pages_dict):
        in_table, match_table_ending, in_figure, in_column = False, False, False, False

        for page_index, pages_content in pages_dict.items():
            if not page_index.isnumeric():
                continue

            tables, figures, columns = [], [], []

            for ind in range(len(pages_content)):
                line = pages_content[ind]
                (_, height, content) = line

                if not (in_table or in_figure or in_column):
                    if re.match('专栏 \\d+ ', content):
                        in_column = True
                        columns.append(ind)

                    elif re.match('表 \\d+ ', content):
                        in_table = True
                        tables.append(ind)
                        match_table_ending = False

                    elif re.match('数据来源：[^。]+?。', content):
                        figures.append(ind)

                        if not re.match('.+?图 \\d+ ', content):
                            in_figure = True
                    continue

                if in_column:
                    if height > self.most_height:
                        in_column = False
                    else:
                        columns.append(ind)
                    continue

                if in_table:
                    if re.match('.*?数据来源：[^。]+?。', content):
                        match_table_ending = True
                        tables.append(ind)
                        continue

                    if match_table_ending and height > self.most_height:
                        in_table = False
                    else:
                        if re.match('表 \\d+ {2}', content):
                            match_table_ending = False
                        tables.append(ind)
                    continue

                if in_figure:
                    if re.match('图 \\d+ {2}', content):
                        in_figure = False
                    figures.append(ind)
                    continue

            cache = [item for item in pages_content if pages_content.index(item) not in (tables + figures + columns)]
            pages_dict[page_index] = cache

        return pages_dict

    def merge_continuous_paragraph(self, pages_dict):
        for page_index, page_content in pages_dict.items():
            if not page_index.isnumeric() or not page_content:
                continue

            cache_page_content = []
            cache_content = ""
            cache_height = []
            cache_x0 = 0

            for ind in range(len(page_content)):
                (x0, height, content) = page_content[ind]

                if abs(x0 - self.most_x0) >= 5:
                    if cache_content:
                        cache_mean_height = sum(cache_height) / len(cache_height)
                        cache_page_content.append((cache_x0, cache_mean_height, cache_content))

                        cache_content = ""
                        cache_height = []

                    cache_content += content
                    cache_height.append(height)
                    cache_x0 = x0
                else:
                    cache_content += content
                    cache_height.append(height)

                    if ind == 0:
                        cache_x0 = x0

            cache_mean_height = sum(cache_height) / len(cache_height)
            cache_page_content.append((cache_x0, cache_mean_height, cache_content))

            pages_dict[page_index] = cache_page_content

        return pages_dict

    def concat_pages_to_plain_text(self, pages_dict):
        plain_text = []

        for page_index, page_content in pages_dict.items():
            if not page_index.isnumeric() or not page_content:
                continue

            for (x0, height, content) in page_content:
                if abs(x0 - self.most_x0) >= 5:
                    plain_text.append([page_index, x0, height, content])
                else:
                    plain_text[-1][2] = (plain_text[-1][1] + height) / 2
                    plain_text[-1][3] += content

        return plain_text

    def get_index_token(self, text):
        pattern = re.compile('第[一二三四五六七八九十]{1,2}部分')
        result = re.match(pattern, text)
        if result:
            return 'A'

        pattern = re.compile('([一二三四五六七八九十]{1,2})([.、])')
        result = re.match(pattern, text)
        if result:
            return 'B'

        pattern = re.compile('[（(][一二三四五六七八九十0-9]{1,2}[)）]')
        result = re.match(pattern, text)
        if result:
            return 'C'

        pattern = re.compile('([0-9]{1,2})([.、])')
        result = re.match(pattern, text)
        if result:
            return 'D'

        return None

    def convert_to_tree(self, plain_text):
        root_node = IndexNode()
        root_node.title = self.pdf_name
        parent_stack = [('root', root_node)]

        for (page_index, x0s, height, content) in plain_text:
            index_token = self.get_index_token(content)

            if len(content) <= self.title_max_length and index_token is not None:
                new_node = IndexNode()
                new_node.page = page_index
                new_node.title = content
                is_added = False

                # 与上一标题行平级
                for i in range(len(parent_stack), 0, -1):
                    ind = i - 1
                    if parent_stack[ind][0] == index_token:
                        while len(parent_stack) > ind:
                            parent_stack.pop()
                        parent_stack[-1][1].children.append(new_node)
                        new_node.parent = parent_stack[-1][1]
                        parent_stack.append((index_token, new_node))
                        is_added = True
                        break

                if not is_added:
                    parent_stack[-1][1].children.append(new_node)
                    new_node.parent = parent_stack[-1][1]
                    parent_stack.append((index_token, new_node))

            else:
                parent_stack[-1][1].paragraphs.append(content)

        return root_node


class IndexNode:
    def __init__(self):
        self.title = ""
        self.paragraphs = []
        self.children = []
        self.parent = "root"
        self.page = 0


class PDFComparer:
    def __init__(self):
        self.pdfs = IndexPairNode()

    def compare_two_pdf(self, pdf_a_path, pdf_b_path, rule, password_a="", password_b=""):
        parser_a = PDFParser()
        parse_result_a = parser_a.parse(pdf_a_path, password_a)
        index_tree_a = parser_a.analyze(rule, parse_result_a)

        parser_b = PDFParser()
        parse_result_b = parser_b.parse(pdf_b_path, password_b)
        index_tree_b = parser_b.analyze(rule, parse_result_b)

        self.pdfs.target = (index_tree_a, index_tree_b)
        self.compare_index_pair_node(self.pdfs)

        return self.pdfs

    def compare_index_pair_node(self, index_pair_node):
        index_pair_node.title_alignment = self.align_title(index_pair_node)
        index_pair_node.paragraphs_alignment = self.align_paragraphs(index_pair_node)
        index_pair_node.children_alignment = self.align_children(index_pair_node)

        if index_pair_node.children_alignment:
            for node in index_pair_node.children_alignment:
                self.compare_index_pair_node(node)

    def align_title(self, index_pair_node):
        title_a = index_pair_node.target[0].title
        title_b = index_pair_node.target[1].title
        alignment = (self.edit_ops(title_a, title_b))

        return alignment

    def align_children(self, index_pair_node):
        children_a = index_pair_node.target[0].children
        children_b = index_pair_node.target[1].children
        alignment = []

        if not children_a and not children_b:

            return alignment

        elif not children_a and children_b:
            for child in children_b:
                node_pair = IndexPairNode()
                node_pair.target = (IndexNode(), child)
                alignment.append(node_pair)

        elif children_a and not children_b:
            for child in children_a:
                node_pair = IndexPairNode()
                node_pair.target = (child, IndexNode())
                alignment.append(node_pair)

        else:
            title_a = [child.title for child in children_a]
            title_b = [child.title for child in children_b]
            alignment_title = self.align_text_list(title_a, title_b)

            for aligned in alignment_title:
                node_a = children_a[title_a.index(aligned[0])] if aligned[0] else IndexNode()
                node_b = children_b[title_b.index(aligned[1])] if aligned[1] else IndexNode()
                node_pair = IndexPairNode()
                node_pair.target = (node_a, node_b)
                alignment.append(node_pair)

        return alignment

    def align_paragraphs(self, index_pair_node):
        paragraphs_a = [re.split("[。？！]", i) for i in index_pair_node.target[0].paragraphs]
        paragraphs_b = [re.split("[。？！]", i) for i in index_pair_node.target[1].paragraphs]
        alignment = []

        for ind in range(len(paragraphs_a)):
            paragraphs_a[ind] = [i for i in paragraphs_a[ind] if i]
        for ind in range(len(paragraphs_b)):
            paragraphs_b[ind] = [i for i in paragraphs_b[ind] if i]

        if not paragraphs_a and not paragraphs_b:

            return alignment

        elif not paragraphs_a and paragraphs_b:
            for para in paragraphs_b:
                cache = []
                for sent in para:
                    cache.append(([], [("insert", sent)]))
                alignment.append(cache)

        elif paragraphs_a and not paragraphs_b:
            for para in paragraphs_a:
                cache = []
                for sent in para:
                    cache.append((["delete", sent], []))
                alignment.append(cache)

        else:
            paragraphs_a = ["。".join(sents) for sents in paragraphs_a]
            paragraphs_b = ["。".join(sents) for sents in paragraphs_b]
            alignment = self.align_text_list(paragraphs_a, paragraphs_b)

            for ind in range(len(alignment)):
                aligned = alignment[ind]
                sent_list_a = aligned[0].split("。") if aligned[0] else []
                sent_list_b = aligned[1].split("。") if aligned[1] else []
                alignment_sents = self.align_text_list(sent_list_a, sent_list_b)

                for i in range(len(alignment_sents)):
                    alignment_sents[i] = (self.edit_ops(alignment_sents[i][0], alignment_sents[i][1]))

                alignment[ind] = alignment_sents

        return alignment

    def align_text_list(self, list_a, list_b):
        distances = []

        for ind_a in range(len(list_a)):
            str_a = list_a[ind_a]
            for ind_b in range(len(list_b)):
                str_b = list_b[ind_b]
                dist = edit.distance(str_a, str_b) / ((len(str_a) + len(str_b)) / 2)
                distances.append((ind_a, ind_b, dist))

        alignment = self.align_distances(distances, list_a, list_b)

        return alignment

    def align_distances(self, distances, list_a, list_b, start_a=0, start_b=0):
        alignment = []
        alignment_cache = []

        if not list_a and not list_b:
            return alignment

        if not list_a and list_b:
            for b in list_b:
                alignment.append(("", b))
            return alignment

        if list_a and not list_b:
            for a in list_a:
                alignment.append((a, ""))
            return alignment

        distances.sort(key=lambda x: x[-1])
        aligned = distances[0]
        ind_a = aligned[0] - start_a
        ind_b = aligned[1] - start_b

        alignment_cache.append((list_a[ind_a], list_b[ind_b], aligned[-1]))
        list_a_left, list_a_right = list_a[:ind_a], list_a[ind_a + 1:]
        list_b_left, list_b_right = list_b[:ind_b], list_b[ind_b + 1:]
        distances_left = [item for item in distances if item[0] < aligned[0] and item[1] < aligned[1]]
        distances_right = [item for item in distances if item[0] > aligned[0] and item[1] > aligned[1]]

        alignment_left = self.align_distances(distances_left, list_a_left, list_b_left, start_a, start_b)
        alignment_right = self.align_distances(distances_right, list_a_right, list_b_right, aligned[0] + 1, aligned[1] + 1)
        alignment_cache = alignment_left + alignment_cache + alignment_right

        for item in alignment_cache:
            if len(item) == 2:
                alignment.append(item)
            elif item[-1] <= 1:
                alignment.append((item[0], item[1]))
            else:
                alignment.append((item[0], ""))
                alignment.append(("", item[1]))

        return alignment

    def edit_ops(self, text_a, text_b):
        ops_a, ops_b = [], []

        if not text_a and not text_b:
            return ops_a, ops_b

        if text_a and not text_b:
            ops_a.append(("delete", text_a))

            return ops_a, ops_b

        if not text_a and text_b:
            ops_b.append(("insert", text_b))

            return ops_a, ops_b

        ops = edit.opcodes(text_a, text_b)
        for op in ops:
            op_name = op[0]
            slice_a = text_a[op[1]: op[2]]
            slice_b = text_b[op[3]: op[4]]

            if slice_a:
                ops_a.append((op_name, slice_a))
            if slice_b:
                ops_b.append((op_name, slice_b))

        return ops_a, ops_b


class IndexPairNode:
    def __init__(self):
        self.target = None
        self.title_alignment = None
        self.paragraphs_alignment = None
        self.children_alignment = None


def main():
    # pdf_input_path = "../Resources/2020Q1.pdf"
    # agent = PDFParser()
    # parse_result = agent.parse(pdf_input_path)
    #
    # with open("result.txt", "w") as f:
    #     for line in parse_result:
    #         f.write(str(line[0]) + '\t' + '%04.1f' % line[1] + '\t' + line[2] + '\n')
    #
    # document = agent.analyze("Monetary Policy Report", parse_result)
    # print(document)
    agent = PDFComparer()
    result = agent.compare_two_pdf("../Resources/2020Q1.pdf", "../Resources/2020Q2.pdf", "Monetary Policy Report")
    print(result)


if __name__ == '__main__':
    main()
