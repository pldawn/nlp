import re
import codecs
import markdown
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
                    if height > min(14, self.most_height) and self.check_chinese(content) and "数据来源" not in content:
                        in_column = False

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

                    else:
                        columns.append(ind)
                    continue

                if in_table:
                    if re.match('.*?数据来源：[^。]+?。', content):
                        match_table_ending = True
                        tables.append(ind)
                        continue

                    if match_table_ending and height > min(14, self.most_height) and self.check_chinese(content):
                        in_table = False

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

                    else:
                        if re.match('表 \\d+ {2}', content):
                            match_table_ending = False
                        tables.append(ind)
                    continue

                if in_figure:
                    if re.match('图 \\d+ {2}', content):
                        in_figure = False

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

                    figures.append(ind)
                    continue

            cache = [item for item in pages_content if pages_content.index(item) not in (tables + figures + columns)]
            pages_dict[page_index] = cache

        return pages_dict

    def check_chinese(self, text):
        if re.search("[\u4e00-\u9fa5]", text):
            return True

        return False

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
            index_token = self.get_index_token(content.replace(" ", ""))

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
                parent_stack[-1][1].paragraphs.append(content.replace(" ", ""))

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
            alignment_title = self.align_text_list_for_children(title_a, title_b)

            for aligned in alignment_title:
                node_a = children_a[title_a.index(aligned[0])] if aligned[0] else IndexNode()
                node_b = children_b[title_b.index(aligned[1])] if aligned[1] else IndexNode()
                node_pair = IndexPairNode()
                node_pair.target = (node_a, node_b)
                alignment.append(node_pair)

        return alignment

    def align_paragraphs(self, index_pair_node):
        paragraphs_a = index_pair_node.target[0].paragraphs
        paragraphs_b = index_pair_node.target[1].paragraphs
        alignment = []

        if not paragraphs_a and not paragraphs_b:

            return alignment

        elif not paragraphs_a and paragraphs_b:
            for para in paragraphs_b:
                alignment.append(([], [("insert", para)]))

        elif paragraphs_a and not paragraphs_b:
            for para in paragraphs_a:
                alignment.append(([("delete", para)], []))

        else:
            alignment_cache = self.align_text_list_for_paragraphs(paragraphs_a, paragraphs_b)

            for ind in range(len(alignment_cache)):
                aligned = alignment_cache[ind]
                sent_list_a = re.split("\\W", paragraphs_a[aligned[0]]) if aligned[0] != 100 else []
                punc_list_a = re.split("\\w", paragraphs_a[aligned[0]])[1:-1] if aligned[0] != 100 else []
                punc_list_a = [i for i in punc_list_a if i]
                sent_list_b = re.split("\\W", paragraphs_b[aligned[1]]) if aligned[1] != 100 else []
                punc_list_b = re.split("\\w", paragraphs_b[aligned[1]])[1:-1] if aligned[1] != 100 else []
                punc_list_b = [i for i in punc_list_b if i]
                alignment_sents = self.align_text_list_for_paragraphs(sent_list_a, sent_list_b)

                alignment_ops = []
                for i in range(len(alignment_sents)):
                    text_a = sent_list_a[alignment_sents[i][0]] if alignment_sents[i][0] != 100 else ""
                    text_b = sent_list_b[alignment_sents[i][1]] if alignment_sents[i][1] != 100 else ""
                    alignment_ops.append((self.edit_ops(text_a, text_b)))

                char_ops_a, char_ops_b = [], []
                for j in range(len(alignment_ops)):
                    if alignment_ops[j][0]:
                        char_ops_a.append((alignment_sents[j][0], alignment_ops[j][0]))
                    if alignment_ops[j][1]:
                        char_ops_b.append((alignment_sents[j][1], alignment_ops[j][1]))

                char_ops_a.sort(key=lambda x: x[0])
                char_ops_b.sort(key=lambda x: x[0])

                assert(len(char_ops_a) == 1 + len(punc_list_a) or (len(char_ops_a) == 0 and len(punc_list_a) == 0))
                assert(len(char_ops_b) == 1 + len(punc_list_b) or (len(char_ops_b) == 0 and len(punc_list_b) == 0))

                ops_a, ops_b = [], []
                for i in range(len(char_ops_a)):
                    ops_a.append(char_ops_a[i][1])
                    if i < len(char_ops_a) - 1:
                        ops_a.append([("equal", punc_list_a[i])])

                for j in range(len(char_ops_b)):
                    ops_b.append(char_ops_b[j][1])
                    if j < len(char_ops_b) - 1:
                        ops_b.append([("equal", punc_list_b[j])])

                res_a, res_b = [], []
                for i in ops_a:
                    res_a += i
                for j in ops_b:
                    res_b += j

                alignment.append((res_a, res_b))

        return alignment

    def align_text_list_for_children(self, list_a, list_b):
        distances = []

        for ind_a in range(len(list_a)):
            str_a = list_a[ind_a]
            for ind_b in range(len(list_b)):
                str_b = list_b[ind_b]
                dist = edit.distance(str_a, str_b) / ((len(str_a) + len(str_b)) / 2)
                distances.append((ind_a, ind_b, dist))

        alignment = self.align_distances_for_children(distances, list_a, list_b)

        return alignment

    def align_distances_for_children(self, distances, list_a, list_b, start_a=0, start_b=0):
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

        alignment_left = self.align_distances_for_children(distances_left, list_a_left, list_b_left, start_a, start_b)
        alignment_right = self.align_distances_for_children(distances_right, list_a_right, list_b_right, aligned[0] + 1, aligned[1] + 1)
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

    def align_text_list_for_paragraphs(self, list_a, list_b):
        distances = []

        for ind_a in range(len(list_a)):
            str_a = list_a[ind_a]
            first_a = str_a.split("。")[0]
            for ind_b in range(len(list_b)):
                str_b = list_b[ind_b]
                first_b = str_b.split("。")[0]
                try:
                    dist1 = edit.distance(str_a, str_b) / ((len(str_a) + len(str_b)) / 2)
                    dist2 = edit.distance(first_a, first_b) / ((len(first_a) + len(first_b)) / 2)
                    dist = min(dist1, dist2)
                except ZeroDivisionError:
                    dist = 0
                distances.append((ind_a, ind_b, dist))

        alignment = self.align_distances_for_paragraphs(distances, list_a, list_b)

        for ind in range(len(list_a)):
            not_in = True
            for item in alignment:
                if ind == item[0]:
                    not_in = False
            if not_in:
                alignment.append((ind, 100))

        for ind in range(len(list_b)):
            not_in = True
            for item in alignment:
                if ind == item[1]:
                    not_in = False
            if not_in:
                alignment.append((100, ind))

        alignment.sort(key=lambda x: x[0])
        alignment.sort(key=lambda x: x[1])

        return alignment

    def align_distances_for_paragraphs(self, distances, list_a, list_b):
        alignment = []
        alignment_cache = []

        if not distances:
            return alignment

        distances.sort(key=lambda x: x[-1])
        aligned = distances[0]
        ind_a = aligned[0]
        ind_b = aligned[1]

        alignment_cache.append((ind_a, ind_b, aligned[-1]))
        distances = [item for item in distances if item[0] != ind_a and item[1] != ind_b]
        alignment_cache = alignment_cache + self.align_distances_for_paragraphs(distances, list_a, list_b)

        for item in alignment_cache:
            if len(item) == 2:
                alignment.append(item)
            elif item[-1] <= 1:
                alignment.append((item[0], item[1]))
            else:
                alignment.append((item[0], 100))
                alignment.append((100, item[1]))

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

            if self.is_numerical(slice_a) and self.is_numerical(slice_b):
                op_name = "equal"

            if slice_a:
                ops_a.append((op_name, slice_a))
            if slice_b:
                ops_b.append((op_name, slice_b))

        return ops_a, ops_b

    def to_markdown_based_frame(self, output_path):
        # if self.pdfs.target is None:
        #     raise AttributeError("haven't call compare_two_pdf interface.")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("<table>\n")

            title1, title2 =  "一季度", "二季度" # "", ""
            # for item in self.pdfs.title_alignment[0]:
            #     title1 += item[1]
            # for item in self.pdfs.title_alignment[1]:
            #     title2 += item[1]
            self.write_to_frame(f, "项目", "", title1, title2)
            self.write_to_frame(f, "总体基调", "", "好", "坏")
            self.write_to_frame(f, "政策展望", "流动性", "A", "B", 2)
            self.write_to_frame(f, "", "风险", "A<br>C", "B")

            f.write("</table>\n")

    def to_markdown(self, output_path):
        if self.pdfs.target is None:
            raise AttributeError("haven't call compare_two_pdf interface.")

        with open(output_path, "w", encoding="utf-8") as f:
            title1, title2 = "", ""
            for item in self.pdfs.title_alignment[0]:
                title1 += item[1]
            for item in self.pdfs.title_alignment[1]:
                title2 += item[1]
            self.write_to_line(f, title1, title2)

            self.write_to_line(f, "----", "----")

            if self.pdfs.paragraphs_alignment:
                for aligned_para in self.pdfs.paragraphs_alignment:
                    str1 = self.decorate(aligned_para[0], True) + "。" if self.decorate(aligned_para[0], True) else ""
                    str2 = self.decorate(aligned_para[1], False) + "。" if self.decorate(aligned_para[1], False) else ""

                    self.write_to_line(f, str1, str2)

            if self.pdfs.children_alignment:
                for aligned_child in self.pdfs.children_alignment:
                    self.node_to_markdown(f, aligned_child)

    def node_to_markdown(self, f, index_pair_node):
        if index_pair_node.title_alignment:
            str1 = self.decorate(index_pair_node.title_alignment[0], True)
            str2 = self.decorate(index_pair_node.title_alignment[1], False)
            self.write_to_line(f, str1, str2)

        if index_pair_node.paragraphs_alignment:
            for aligned_para in index_pair_node.paragraphs_alignment:
                str1 = self.decorate(aligned_para[0], True) + "。" if self.decorate(aligned_para[0], True) else ""
                str2 = self.decorate(aligned_para[1], False) + "。" if self.decorate(aligned_para[1], False) else ""

                self.write_to_line(f, str1, str2)

        if index_pair_node.children_alignment:
            for aligned_child in index_pair_node.children_alignment:
                self.node_to_markdown(f, aligned_child)

    def write_to_frame(self, f, str1, str2, str3, str4, rowspan1=1, rowspan2=1):
        if str1 and str2 and str3 and str4:
            f.write(" <tr>\n")
            f.write("  <td rowspan='%s'>%s</td>\n" % (rowspan1, str1))
            f.write("  <td rowspan='%s'>%s</td>\n" % (rowspan2, str2))
            f.write("  <td>%s</td>\n" % str3)
            f.write("  <td>%s</td>\n" % str4)
            f.write(" </tr>\n")

        elif str1 and not str2 and str3 and str4:
            f.write(" <tr>\n")
            f.write("  <td colspan='2'>%s</td>\n" % str1)
            f.write("  <td>%s</td>\n" % str3)
            f.write("  <td>%s</td>\n" % str4)
            f.write(" </tr>\n")

        elif not str1 and str2 and str3 and str4:
            f.write(" <tr>\n")
            f.write("  <td rowspan='%s'>%s</td>\n" % (rowspan2, str2))
            f.write("  <td>%s</td>\n" % str3)
            f.write("  <td>%s</td>\n" % str4)
            f.write(" </tr>\n")

        elif not str1 and not str2 and str3 and str4:
            f.write(" <tr>\n")
            f.write("  <td>%s</td>\n" % str3)
            f.write("  <td>%s</td>\n" % str4)
            f.write(" </tr>\n")

        else:
            raise ValueError("write_to_frame")

    def write_to_line(self, f, str1, str2):
        f.write("|" + str1 + "|" + str2 + "|" + "\n")

    def decorate(self, ops, source, delete_color="#FF69B4", insert_color="#008000"):
        string = ""
        for op in ops:
            name = op[0]
            text = op[1]

            if name == "equal":
                string += text

            elif name == "replace":
                if source:
                    text = ("<font color=%s>" % delete_color) + text + "</font>"
                else:
                    text = ("<font color=%s>" % insert_color) + text + "</font>"
                string += text

            elif name == "delete":
                text = ("<font color=%s>" % delete_color) + text + "</font>"
                string += text

            elif name == "insert":
                text = ("<font color=%s>" % insert_color) + text + "</font>"
                string += text

            else:
                string += text

        return string

    def is_numerical(self, text):
        try:
            float(text)
            return True
        except ValueError:
            return False


class IndexPairNode:
    def __init__(self):
        self.target = None
        self.title_alignment = None
        self.paragraphs_alignment = None
        self.children_alignment = None


def markdown_to_html(input_path, output_path):
    input_file = codecs.open(input_path, mode="r", encoding="utf-8")
    text = input_file.read()

    html = markdown.markdown(text, extensions=['markdown.extensions.tables'])
    css = \
        """
        <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
        <style type="text/css">
            body{
                margin: 0 auto;
                font-family: "Microsoft YaHei", arial,sans-serif;
                color: #444444;
                line-height: 1;
                padding: 30px;
            }
            @media screen and (min-width: 768px) {
                body {
                    width: 748px;
                    margin: 10px auto;
                }
            }
            h1, h2, h3, h4 {
                color: #111111;
                font-weight: 400;
                margin-top: 1em;
            }
            
            h1, h2, h3, h4, h5 {
                font-family: Georgia, Palatino, serif;
            }
            h1, h2, h3, h4, h5, p , dl{
                margin-bottom: 16px;
                padding: 0;
            }
            h1 {
                font-size: 48px;
                line-height: 54px;
            }
            h2 {
                font-size: 36px;
                line-height: 42px;
            }
            h1, h2 {
                border-bottom: 1px solid #EFEAEA;
                padding-bottom: 10px;
            }
            h3 {
                font-size: 24px;
                line-height: 30px;
            }
            h4 {
                font-size: 21px;
                line-height: 26px;
            }
            h5 {
                font-size: 18px;
                list-style: 23px;
            }
            a {
                color: #0099ff;
                margin: 0;
                padding: 0;
                vertical-align: baseline;
            }
            a:hover {
                text-decoration: none;
                color: #ff6600;
            }
            a:visited {
                /*color: purple;*/
            }
            ul, ol {
                padding: 0;
                padding-left: 24px;
                margin: 0;
            }
            li {
                line-height: 24px;
            }
            p, ul, ol {
                font-size: 16px;
                line-height: 24px;
            }
            
            ol ol, ul ol {
                list-style-type: lower-roman;
            }
            
            /*pre {
                padding: 0px 24px;
                max-width: 800px;
                white-space: pre-wrap;
            }
            code {
                font-family: Consolas, Monaco, Andale Mono, monospace;
                line-height: 1.5;
                font-size: 13px;
            }*/
            
            code, pre {
                border-radius: 3px;
                background-color:#f7f7f7;
                color: inherit;
            }
            
            code {
                font-family: Consolas, Monaco, Andale Mono, monospace;
                margin: 0 2px;
            }
            
            pre {
                line-height: 1.7em;
                overflow: auto;
                padding: 6px 10px;
                border-left: 5px solid #6CE26C;
            }
            
            pre > code {
                border: 0;
                display: inline;
                max-width: initial;
                padding: 0;
                margin: 0;
                overflow: initial;
                line-height: inherit;
                font-size: .85em;
                white-space: pre;
                background: 0 0;
            
            }
            
            code {
                color: #666555;
            }
            
            
            /** markdown preview plus 对于代码块的处理有些问题, 所以使用统一的颜色 */
            /*code .keyword {
              color: #8959a8;
            }
            
            code .number {
              color: #f5871f;
            }
            
            code .comment {
              color: #998
            }*/
            
            aside {
                display: block;
                float: right;
                width: 390px;
            }
            blockquote {
                border-left:.5em solid #eee;
                padding: 0 0 0 2em;
                margin-left:0;
            }
            blockquote  cite {
                font-size:14px;
                line-height:20px;
                color:#bfbfbf;
            }
            blockquote cite:before {
                content: '\2014 \00A0';
            }
            
            blockquote p {
                color: #666;
            }
            hr {
                text-align: left;
                color: #999;
                height: 2px;
                padding: 0;
                margin: 16px 0;
                background-color: #e7e7e7;
                border: 0 none;
            }
            
            dl {
                padding: 0;
            }
            
            dl dt {
                padding: 10px 0;
                margin-top: 16px;
                font-size: 1em;
                font-style: italic;
                font-weight: bold;
            }
            
            dl dd {
                padding: 0 16px;
                margin-bottom: 16px;
            }
            
            dd {
                margin-left: 0;
            }
            
            /* Code below this line is copyright Twitter Inc. */
            
            button,
            input,
            select,
            textarea {
                font-size: 100%;
                margin: 0;
                vertical-align: baseline;
                *vertical-align: middle;
            }
            button, input {
                line-height: normal;
                *overflow: visible;
            }
            button::-moz-focus-inner, input::-moz-focus-inner {
                border: 0;
                padding: 0;
            }
            button,
            input[type="button"],
            input[type="reset"],
            input[type="submit"] {
                cursor: pointer;
                -webkit-appearance: button;
            }
            input[type=checkbox], input[type=radio] {
                cursor: pointer;
            }
            /* override default chrome & firefox settings */
            input:not([type="image"]), textarea {
                -webkit-box-sizing: content-box;
                -moz-box-sizing: content-box;
                box-sizing: content-box;
            }
            
            input[type="search"] {
                -webkit-appearance: textfield;
                -webkit-box-sizing: content-box;
                -moz-box-sizing: content-box;
                box-sizing: content-box;
            }
            input[type="search"]::-webkit-search-decoration {
                -webkit-appearance: none;
            }
            label,
            input,
            select,
            textarea {
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
                font-size: 13px;
                font-weight: normal;
                line-height: normal;
                margin-bottom: 18px;
            }
            input[type=checkbox], input[type=radio] {
                cursor: pointer;
                margin-bottom: 0;
            }
            input[type=text],
            input[type=password],
            textarea,
            select {
                display: inline-block;
                width: 210px;
                padding: 4px;
                font-size: 13px;
                font-weight: normal;
                line-height: 18px;
                height: 18px;
                color: #808080;
                border: 1px solid #ccc;
                -webkit-border-radius: 3px;
                -moz-border-radius: 3px;
                border-radius: 3px;
            }
            select, input[type=file] {
                height: 27px;
                line-height: 27px;
            }
            textarea {
                height: auto;
            }
            /* grey out placeholders */
            :-moz-placeholder {
                color: #bfbfbf;
            }
            ::-webkit-input-placeholder {
                color: #bfbfbf;
            }
            input[type=text],
            input[type=password],
            select,
            textarea {
                -webkit-transition: border linear 0.2s, box-shadow linear 0.2s;
                -moz-transition: border linear 0.2s, box-shadow linear 0.2s;
                transition: border linear 0.2s, box-shadow linear 0.2s;
                -webkit-box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
                -moz-box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
                box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
            }
            input[type=text]:focus, input[type=password]:focus, textarea:focus {
                outline: none;
                border-color: rgba(82, 168, 236, 0.8);
                -webkit-box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1), 0 0 8px rgba(82, 168, 236, 0.6);
                -moz-box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1), 0 0 8px rgba(82, 168, 236, 0.6);
                box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1), 0 0 8px rgba(82, 168, 236, 0.6);
            }
            /* buttons */
            button {
                display: inline-block;
                padding: 4px 14px;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
                font-size: 13px;
                line-height: 18px;
                -webkit-border-radius: 4px;
                -moz-border-radius: 4px;
                border-radius: 4px;
                -webkit-box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.2), 0 1px 2px rgba(0, 0, 0, 0.05);
                -moz-box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.2), 0 1px 2px rgba(0, 0, 0, 0.05);
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.2), 0 1px 2px rgba(0, 0, 0, 0.05);
                background-color: #0064cd;
                background-repeat: repeat-x;
                background-image: -khtml-gradient(linear, left top, left bottom, from(#049cdb), to(#0064cd));
                background-image: -moz-linear-gradient(top, #049cdb, #0064cd);
                background-image: -ms-linear-gradient(top, #049cdb, #0064cd);
                background-image: -webkit-gradient(linear, left top, left bottom, color-stop(0%, #049cdb), color-stop(100%, #0064cd));
                background-image: -webkit-linear-gradient(top, #049cdb, #0064cd);
                background-image: -o-linear-gradient(top, #049cdb, #0064cd);
                background-image: linear-gradient(top, #049cdb, #0064cd);
                color: #fff;
                text-shadow: 0 -1px 0 rgba(0, 0, 0, 0.25);
                border: 1px solid #004b9a;
                border-bottom-color: #003f81;
                -webkit-transition: 0.1s linear all;
                -moz-transition: 0.1s linear all;
                transition: 0.1s linear all;
                border-color: #0064cd #0064cd #003f81;
                border-color: rgba(0, 0, 0, 0.1) rgba(0, 0, 0, 0.1) rgba(0, 0, 0, 0.25);
            }
            button:hover {
                color: #fff;
                background-position: 0 -15px;
                text-decoration: none;
            }
            button:active {
                -webkit-box-shadow: inset 0 3px 7px rgba(0, 0, 0, 0.15), 0 1px 2px rgba(0, 0, 0, 0.05);
                -moz-box-shadow: inset 0 3px 7px rgba(0, 0, 0, 0.15), 0 1px 2px rgba(0, 0, 0, 0.05);
                box-shadow: inset 0 3px 7px rgba(0, 0, 0, 0.15), 0 1px 2px rgba(0, 0, 0, 0.05);
            }
            button::-moz-focus-inner {
                padding: 0;
                border: 0;
            }
            table {
                *border-collapse: collapse; /* IE7 and lower */
                border-spacing: 0;
                width: 100%;
            }
            table {
                border: solid #ccc 1px;
                -moz-border-radius: 6px;
                -webkit-border-radius: 6px;
                border-radius: 6px;
                /*-webkit-box-shadow: 0 1px 1px #ccc;
                -moz-box-shadow: 0 1px 1px #ccc;
                box-shadow: 0 1px 1px #ccc;   */
            }
            table tr:hover {
                background: #fbf8e9;
                -o-transition: all 0.1s ease-in-out;
                -webkit-transition: all 0.1s ease-in-out;
                -moz-transition: all 0.1s ease-in-out;
                -ms-transition: all 0.1s ease-in-out;
                transition: all 0.1s ease-in-out;
            }
            table td, .table th {
                border-left: 1px solid #ccc;
                border-top: 1px solid #ccc;
                padding: 10px;
                text-align: left;
            }
            
            table th {
                background-color: #dce9f9;
                background-image: -webkit-gradient(linear, left top, left bottom, from(#ebf3fc), to(#dce9f9));
                background-image: -webkit-linear-gradient(top, #ebf3fc, #dce9f9);
                background-image:    -moz-linear-gradient(top, #ebf3fc, #dce9f9);
                background-image:     -ms-linear-gradient(top, #ebf3fc, #dce9f9);
                background-image:      -o-linear-gradient(top, #ebf3fc, #dce9f9);
                background-image:         linear-gradient(top, #ebf3fc, #dce9f9);
                /*-webkit-box-shadow: 0 1px 0 rgba(255,255,255,.8) inset;
                -moz-box-shadow:0 1px 0 rgba(255,255,255,.8) inset;
                box-shadow: 0 1px 0 rgba(255,255,255,.8) inset;*/
                border-top: none;
                text-shadow: 0 1px 0 rgba(255,255,255,.5);
                padding: 5px;
            }
            
            table td:first-child, table th:first-child {
                border-left: none;
            }
            
            table th:first-child {
                -moz-border-radius: 6px 0 0 0;
                -webkit-border-radius: 6px 0 0 0;
                border-radius: 6px 0 0 0;
            }
            table th:last-child {
                -moz-border-radius: 0 6px 0 0;
                -webkit-border-radius: 0 6px 0 0;
                border-radius: 0 6px 0 0;
            }
            table th:only-child{
                -moz-border-radius: 6px 6px 0 0;
                -webkit-border-radius: 6px 6px 0 0;
                border-radius: 6px 6px 0 0;
            }
            table tr:last-child td:first-child {
                -moz-border-radius: 0 0 0 6px;
                -webkit-border-radius: 0 0 0 6px;
                border-radius: 0 0 0 6px;
            }
            table tr:last-child td:last-child {
                -moz-border-radius: 0 0 6px 0;
                -webkit-border-radius: 0 0 6px 0;
                border-radius: 0 0 6px 0;
            }
        </style>
        """

    output_file = codecs.open(output_path, mode="w", encoding="utf-8")
    output_file.write(css + html)


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
    # result = agent.compare_two_pdf("../Resources/2020Q1.pdf", "../Resources/2020Q2.pdf", "Monetary Policy Report")
    # agent.to_markdown("result.md")
    # markdown_to_html("result.md", "result.html")
    agent.to_markdown_based_frame("result_frame.md")


if __name__ == '__main__':
    main()
