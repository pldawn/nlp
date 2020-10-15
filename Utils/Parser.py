import re
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
            analyzer = MonetaryPolicyReportAnalyzer()
            result = analyzer.analyze(parse_result)

        else:
            print("rule is invalid, doesn't analyze content")
            return parse_result

        return result


class MonetaryPolicyReportAnalyzer:
    def __init__(self):
        self.pages = OrderedDict()
        self.indices = OrderedDict()
        self.index_to_page = OrderedDict()

    def analyze(self, parse_result):
        self.pages = self.divide_to_pages(parse_result)
        self.pages = self.delete_non_text_part(self.pages)

        return self.pages

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
        in_table, in_figure, in_column = False, False, False

        for page_index, pages_content in pages_dict.items():
            if not page_index.isnumeric():
                continue

            tables, figures, columns = [], [], []

            for ind in range(len(pages_content)):
                line = pages_content[ind]
                (_, height, content) = line

                if not (in_table or in_figure or in_column):
                    if re.match('专栏 \\d+ {2}', content):
                        in_column = True
                        columns.append(ind)

                    elif re.match('表 \\d+ {2}', content):
                        in_table = True
                        tables.append(ind)

                    elif re.match('数据来源：[^。]+?。', content):
                        figures.append(ind)

                        if not re.match('.+?图 \\d+ {2}', content):
                            in_figure = True

                if in_column:
                    if ind == 0:
                        columns.append(ind)
                    else:
                        if height - pages_content[ind-1][1] > 2:
                            in_column = False
                        else:
                            columns.append(ind)
                    continue

                if in_table:
                    if ind == 0:
                        tables.append(ind)
                    else:
                        if re.match('.*?数据来源：[^。]+?。', content):
                            in_table = False
                        tables.append(ind)
                    continue

                if in_figure:
                    if ind == 0:
                        figures.append(ind)
                    else:
                        if re.match('图 \\d+ {2}', content):
                            in_figure = False
                        figures.append(ind)
                    continue

            cache = [item for item in pages_content if pages_content.index(item) not in (tables + figures + columns)]
            pages_dict[page_index] = cache

        return pages_dict


def main():
    pdf_input_path = "/Users/zhangwentao/Documents/中信/项目/金融资讯分析/货币政策执行报告 -输入包+输出包/货币政策执行报告 -输入包/2020Q2.pdf"
    agent = PDFParser()
    result = agent.parse(pdf_input_path)

    # with open("result.txt", "w") as f:
    #     for line in result:
    #         f.write(str(line[0] + '\t' + '%04.1f' % line[1] + '\t' + line[2] + '\n'))

    pages = agent.analyze("Monetary Policy Report", result)
    print(pages)


if __name__ == '__main__':
    main()
