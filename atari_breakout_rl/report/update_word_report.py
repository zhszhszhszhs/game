"""Insert the measured report figures into the existing DOCX, preserving styles.

Uses only the Python standard library. Existing cover, equations, tables and
diagrams remain in the document. The update can be repeated after rebuilding PNGs.
"""
from pathlib import Path
import re
import struct
import zipfile
from xml.dom import minidom

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'report/基于DQN的Breakout智能体课程报告.docx'
IMAGES = ROOT / 'report/latex/images'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
REL = 'http://schemas.openxmlformats.org/package/2006/relationships'


def text_of(node):
    return ''.join(t.firstChild.data if t.firstChild else '' for t in node.getElementsByTagName('w:t'))


def set_text(node, text):
    props = node.getElementsByTagName('w:rPr')
    props = props[0].cloneNode(True) if props else None
    for child in list(node.childNodes):
        if child.nodeName != 'w:pPr':
            node.removeChild(child)
    run = node.ownerDocument.createElementNS(W, 'w:r')
    if props is not None:
        run.appendChild(props)
    element = node.ownerDocument.createElementNS(W, 'w:t')
    element.appendChild(node.ownerDocument.createTextNode(text))
    run.appendChild(element)
    node.appendChild(run)


def preceding_paragraph(node):
    candidate = node.previousSibling
    while candidate is not None and candidate.nodeName != 'w:p':
        candidate = candidate.previousSibling
    if candidate is None:
        raise ValueError('No image paragraph before caption')
    return candidate


def set_dimensions(picture, data):
    width, height = struct.unpack('>II', data[16:24])
    cx = 5760720  # 6.3 inches, within the existing document margins.
    cy = round(cx * height / width)
    for name in ['wp:extent', 'a:ext']:
        for element in picture.getElementsByTagName(name):
            element.setAttribute('cx', str(cx))
            element.setAttribute('cy', str(cy))


def main():
    with zipfile.ZipFile(PATH) as source:
        files = {item.filename: source.read(item.filename) for item in source.infolist()}
    document = minidom.parseString(files['word/document.xml'])
    relationships = minidom.parseString(files['word/_rels/document.xml.rels'])
    captions = {text_of(p): p for p in document.getElementsByTagName('w:p') if text_of(p).startswith('图')}

    def find_prefix(prefix):
        if prefix.startswith('图'):
            description = re.sub(r'^图\d+\s*', '', prefix)
            return next(p for p in document.getElementsByTagName('w:p')
                        if text_of(p).startswith('图')
                        and re.sub(r'^图\d+\s*', '', text_of(p)).startswith(description))
        return next(p for p in document.getElementsByTagName('w:p') if text_of(p).startswith(prefix))

    def replace_image(caption_prefix, filename, new_caption):
        caption = find_prefix(caption_prefix)
        picture = preceding_paragraph(caption)
        rid = picture.getElementsByTagName('a:blip')[0].getAttributeNS(R, 'embed')
        relation = next(e for e in relationships.getElementsByTagName('Relationship') if e.getAttribute('Id') == rid)
        data = (IMAGES / filename).read_bytes()
        files['word/' + relation.getAttribute('Target')] = data
        set_dimensions(picture, data)
        set_text(caption, new_caption)
        return picture, caption

    already_updated = any('Breakout网页实际运行界面' in t for t in captions)
    image_template, caption_template = replace_image(
        '图3  共享环境工厂' if already_updated else '图2  连续画面',
        'observation_pipeline.png', '图3  共享环境工厂中的真实RGB、最大池化结果与四帧灰度输入')
    replace_image('图5  四阶段实际' if already_updated else '图4  四个训练阶段',
                  'training_curves.png', '图5  四阶段实际训练回合奖励与开发评估曲线（星号为最佳节点）')
    replace_image('图6  同一组30个' if already_updated else '图5  同一组30个',
                  'score_distribution.png', '图6  同一组30个开发种子的真实配对得分与分布')
    if not already_updated:
        set_text(find_prefix('图3  训练 评估'), '图4  训练 评估 选模与模型保存构成完整工程流程')

    def insert_image(after, filename, caption_text, rid, image_id):
        picture = image_template.cloneNode(True)
        picture.getElementsByTagName('a:blip')[0].setAttributeNS(R, 'r:embed', rid)
        for name in ['wp:docPr', 'pic:cNvPr']:
            for element in picture.getElementsByTagName(name):
                element.setAttribute('id', str(image_id))
                element.setAttribute('name', filename)
                element.setAttribute('descr', caption_text)
        data = (IMAGES / filename).read_bytes()
        set_dimensions(picture, data)
        caption = caption_template.cloneNode(True)
        set_text(caption, caption_text)
        body = after.parentNode
        following = after.nextSibling
        body.insertBefore(picture, following)
        body.insertBefore(caption, following)
        target = 'media/report_' + filename
        files['word/' + target] = data
        relation = relationships.createElementNS(REL, 'Relationship')
        relation.setAttribute('Id', rid)
        relation.setAttribute('Type', R + '/image')
        relation.setAttribute('Target', target)
        relationships.documentElement.appendChild(relation)
        return caption

    if already_updated:
        replace_image('图2  Breakout网页', 'game_live.png',
                      '图2  Breakout网页实际运行界面：种子42000、得分20、剩余5条生命')
        replace_image('图7  同一局', 'behavior_sequence.png',
                      '图7  同一局中的成功接球与失球连续帧（种子42000，采样间隔0.2秒）')
    else:
        insert_image(find_prefix('该环境是对规则的工程重建'), 'game_live.png',
                     '图2  Breakout网页实际运行界面：种子42000、得分20、剩余5条生命', 'rIdReportGame', 101)
        caption = insert_image(find_prefix('开发评估和网页演示共同说明'), 'behavior_sequence.png',
                               '图7  同一局中的成功接球与失球连续帧（种子42000，采样间隔0.2秒）',
                               'rIdReportBehavior', 102)
        analysis = find_prefix('开发评估和网页演示共同说明').cloneNode(True)
        set_text(analysis, '图7取自推荐最佳模型的同一局游戏。上排小球下降时挡板向接触位置移动，随后小球反弹向上；'
                 '下排小球靠近右边界时超出挡板覆盖范围，落到挡板下方，生命由5减为4。'
                 '画面裁取y=135至210区域，黄圈标注小球，蓝箭头表示相邻采样间的挡板位移。'
                 '单个成功片段不能证明模型具备精确落点预测能力，这些帧也不足以解释所有失球原因。')
        caption.parentNode.insertBefore(analysis, caption.nextSibling)

    for p in document.getElementsByTagName('w:p'):
        text = text_of(p)
        if '对首尾帧做最大池化' in text:
            set_text(p, text.replace('对首尾帧做最大池化', '对实际产生的最后两帧做最大池化'))
        if text.startswith('在42000至42029的30个开发种子上'):
            set_text(p, '在42000至42029的30个开发种子上，本次固定已有权重并重新运行CPU贪心推理。'
                     '推荐最佳模型的均分为260.27、中位数为257.00，续训最终模型的均分为122.40、中位数为101.50，'
                     '与原记录的均分一致。最佳模型在28/30局胜出，平均配对差为137.87分。'
                     '图6直接使用逐局实测分数，原始评估CSV保存在results/evaluation/report_best_30及report_final_30，'
                     '配对表保存在report/data/paired_scores.csv。这些种子仍属于开发数据，不能代替最终留出测试。')

    normal_template = find_prefix('原始观测为').cloneNode(True)
    heading_template = find_prefix('基础训练配置').cloneNode(True)

    def add_text(before, text, heading=False):
        paragraph = (heading_template if heading else normal_template).cloneNode(True)
        set_text(paragraph, text)
        before.parentNode.insertBefore(paragraph, before)
        return paragraph

    diagnostics_caption = '图5  探索率与损失日志诊断：红虚线标记开发最佳节点'
    if any('探索率与损失日志诊断' in text_of(p) and text_of(p).startswith('图')
           for p in document.getElementsByTagName('w:p')):
        replace_image('图5  探索率与损失日志诊断', 'training_diagnostics.png', diagnostics_caption)
    else:
        anchor = find_prefix('六 实验结果')
        add_text(anchor, '探索率与损失日志诊断', heading=True)
        add_text(anchor, '补充图直接读取基础训练和低学习率续训的progress.csv。损失缺失值不插值，'
                 '平滑曲线采用相邻20个有效日志点的滚动中位数；日志间隔不完全一致，'
                 '因此该窗口不代表固定的决策步数。预热期间尚未更新参数，损失曲线保留为空白。')
        add_text(anchor, '基础训练的探索率在300k降至0.05，对应开发均分173.55；350k为174.00。'
                 '此后探索率保持不变，400k和450k的开发均分仍分别回落至70.20与68.10，'
                 '说明后期回退不能仅用探索率继续下降解释。续训探索率缓慢下降，最佳结果却出现在25k，'
                 '探索安排与策略得分之间同样没有简单的单调关系。')
        last = add_text(anchor, '基础训练20k至100k的损失日志点中位数约为0.000425，400k至500k约为0.0402。'
                        '早期较小的拟合误差并未对应较强控制能力。回放样本与TD目标会随策略变化，'
                        '损失需要结合经验分布理解；模型选择应依据独立开发得分，日志不足以单独确定回退原因。')
        insert_image(last, 'training_diagnostics.png', diagnostics_caption, 'rIdReportDiagnostics', 103)

    robustness_caption = '图8  低分风险与排序配对差：经验累积分布和逐种子收益'
    if any('低分风险与排序配对差' in text_of(p) and text_of(p).startswith('图')
           for p in document.getElementsByTagName('w:p')):
        replace_image('图8  低分风险与排序配对差', 'score_robustness.png', robustness_caption)
    else:
        anchor = find_prefix('七 结果分析')
        add_text(anchor, '低分风险与配对差', heading=True)
        add_text(anchor, '在相同30个开发种子上，最佳模型低于50分的局数为1/30（3.33%），'
                 '续训最终模型为9/30（30.00%），相差26.67个百分点。'
                 '第25百分位得分分别为210.00和36.75，说明提升也体现在低分尾部。'
                 '50分是描述性比较阈值，不是环境终止条件；分位数采用线性插值。')
        table = document.createElementNS(W, 'w:tbl')
        props = document.createElementNS(W, 'w:tblPr')
        width = document.createElementNS(W, 'w:tblW')
        width.setAttribute('w:w', '9242')
        width.setAttribute('w:type', 'dxa')
        props.appendChild(width)
        borders = document.createElementNS(W, 'w:tblBorders')
        for edge in ['top', 'bottom', 'insideH']:
            border = document.createElementNS(W, f'w:{edge}')
            border.setAttribute('w:val', 'single')
            border.setAttribute('w:sz', '4')
            border.setAttribute('w:color', 'CCD5DF')
            borders.appendChild(border)
        props.appendChild(borders)
        table.appendChild(props)
        grid = document.createElementNS(W, 'w:tblGrid')
        for size in [4242, 2500, 2500]:
            column = document.createElementNS(W, 'w:gridCol')
            column.setAttribute('w:w', str(size))
            grid.appendChild(column)
        table.appendChild(grid)
        rows = [
            ['指标', '推荐最佳模型', '续训最终模型'],
            ['得分标准差（总体形式）', '103.75', '90.63'],
            ['第10百分位得分', '111.90', '19.90'],
            ['第25百分位得分', '210.00', '36.75'],
            ['低于50分的局数', '1/30', '9/30'],
            ['低于50分的比例', '3.33%', '30.00%'],
            ['最低分至最高分', '36至444', '3至345'],
            ['时间截断局数', '0', '0'],
            ['完整通关局数（864分）', '0', '0'],
        ]
        for values in rows:
            row = document.createElementNS(W, 'w:tr')
            for value in values:
                cell = document.createElementNS(W, 'w:tc')
                paragraph = normal_template.cloneNode(True)
                set_text(paragraph, value)
                ppr = paragraph.getElementsByTagName('w:pPr')[0]
                for name in ['w:ind', 'w:spacing']:
                    for child in list(ppr.getElementsByTagName(name)):
                        child.parentNode.removeChild(child)
                cell.appendChild(paragraph)
                row.appendChild(cell)
            table.appendChild(row)
        anchor.parentNode.insertBefore(table, anchor)
        add_text(anchor, '最佳模型的绝对标准差仍达到103.75，且两模型均没有完整通关。'
                 '因此，低分风险下降仍伴随局间波动，不能表述为稳定通关。')
        last = add_text(anchor, '逐种子配对差显示最佳模型胜出28/30局，但在种子42001和42013上分别少得198分和9分。'
                        '均值差为137.87分，改善未覆盖每个开局。按完整种子对重采样20000次，'
                        '百分位Bootstrap 95%区间为[95.77,177.43]分，随机数种子为20261017。'
                        '该区间仅描述固定两模型在当前开发样本上的配对差变异，不覆盖独立重复训练的变异，'
                        '也不替代最终留出测试。')
        insert_image(last, 'score_robustness.png', robustness_caption, 'rIdReportRobustness', 104)

    body = document.getElementsByTagName('w:body')[0]
    removing = False
    for child in list(body.childNodes):
        if child.nodeName == 'w:p' and text_of(child) == '参考文献':
            removing = True
        if removing and child.nodeName != 'w:sectPr':
            body.removeChild(child)

    number = 0
    figure_numbers = {}
    for p in document.getElementsByTagName('w:p'):
        text = text_of(p)
        if re.match(r'^图\d+\s+', text):
            previous = preceding_paragraph(p)
            if not previous.getElementsByTagName('a:blip'):
                continue
            number += 1
            description = re.sub(r'^图\d+\s*', '', text)
            set_text(p, f'图{number}  {description}')
            figure_numbers[description] = number
    distribution_number = next(n for d, n in figure_numbers.items() if d.startswith('同一组30个'))
    behavior_number = next(n for d, n in figure_numbers.items() if d.startswith('同一局中的成功接球'))
    for p in document.getElementsByTagName('w:p'):
        text = text_of(p)
        if text.startswith('在42000至42029的30个开发种子上'):
            set_text(p, text.replace('图6直接使用', f'图{distribution_number}直接使用'))
        elif re.match(r'^图\d+取自推荐最佳模型', text):
            set_text(p, re.sub(r'^图\d+', f'图{behavior_number}', text))

    files['word/document.xml'] = document.toxml(encoding='UTF-8')
    files['word/_rels/document.xml.rels'] = relationships.toxml(encoding='UTF-8')
    temporary = PATH.with_suffix('.tmp.docx')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as result:
        for name, data in files.items():
            result.writestr(name, data)
    temporary.replace(PATH)
    print(f'Updated {PATH}')


if __name__ == '__main__':
    main()
