"""Add PPO within the original Word outline, then append the method comparison.

Generated blocks are tagged Word content controls so rerunning replaces them.
Native Office Math is used for equations; existing report styles are reused.
"""
import json
from pathlib import Path

W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M='http://schemas.openxmlformats.org/officeDocument/2006/math'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
ROOT=Path(__file__).resolve().parents[1]


def cleanup(document, relationships):
    body=document.getElementsByTagName('w:body')[0]
    for node in list(body.childNodes):
        if node.nodeName=='w:sdt' and any(t.getAttribute('w:val').startswith('ppo-report:')
                for t in node.getElementsByTagName('w:tag')):
            body.removeChild(node)
    for node in list(relationships.documentElement.childNodes):
        if node.nodeType==node.ELEMENT_NODE and node.getAttribute('Id').startswith('rIdPPOReport'):
            relationships.documentElement.removeChild(node)


def supplement(document, files, find_prefix, set_text, insert_image):
    data=json.loads((ROOT/'report/data/ppo_report_summary.json').read_text())
    metrics=data['metrics']; pair=data['paired']
    normal=find_prefix('原始观测为')
    h1=find_prefix('四 系统实现')
    h2=find_prefix('基础训练配置')
    body=document.getElementsByTagName('w:body')[0]
    document.documentElement.setAttribute('xmlns:m',M)
    def element(prefix,name):
        return document.createElementNS(M if prefix=='m' else W,prefix+':'+name)
    def p(text,heading=0):
        node=(h1 if heading==1 else h2 if heading==2 else normal).cloneNode(True)
        set_text(node,text)
        return node
    def block(name,before=None):
        control=element('w','sdt');props=element('w','sdtPr')
        tag=element('w','tag');tag.setAttribute('w:val','ppo-report:'+name)
        props.appendChild(tag);control.appendChild(props)
        content=element('w','sdtContent');control.appendChild(content)
        body.insertBefore(control,before if before is not None else body.getElementsByTagName('w:sectPr')[-1])
        return content
    def text(parent,value,heading=0):parent.appendChild(p(value,heading))
    def table(parent,rows):
        node=element('w','tbl');props=element('w','tblPr')
        width=element('w','tblW');width.setAttribute('w:w','9242');width.setAttribute('w:type','dxa')
        props.appendChild(width)
        layout=element('w','tblLayout');layout.setAttribute('w:type','fixed');props.appendChild(layout)
        borders=element('w','tblBorders')
        for edge in ['top','bottom','insideH']:
            border=element('w',edge);border.setAttribute('w:val','single')
            border.setAttribute('w:sz','4');border.setAttribute('w:color','CCD5DF')
            borders.appendChild(border)
        props.appendChild(borders);node.appendChild(props)
        grid=element('w','tblGrid')
        n=len(rows[0]);sizes=[9242//n]*n
        for size in sizes:
            col=element('w','gridCol');col.setAttribute('w:w',str(size));grid.appendChild(col)
        node.appendChild(grid)
        for index,values in enumerate(rows):
            row=element('w','tr')
            if index==0:
                pr=element('w','trPr');pr.appendChild(element('w','tblHeader'));row.appendChild(pr)
            for col_index,value in enumerate(values):
                cell=element('w','tc');para=p(str(value))
                tcpr=element('w','tcPr');tcw=element('w','tcW')
                tcw.setAttribute('w:w',str(sizes[col_index]));tcw.setAttribute('w:type','dxa')
                tcpr.appendChild(tcw);cell.appendChild(tcpr)
                for pr in para.getElementsByTagName('w:pPr'):
                    for child in list(pr.childNodes):
                        if child.nodeName in ['w:ind','w:spacing']:pr.removeChild(child)
                for run in para.getElementsByTagName('w:r'):
                    rp=run.getElementsByTagName('w:rPr')
                    rp=rp[0] if rp else element('w','rPr')
                    if rp.parentNode is None:run.insertBefore(rp,run.firstChild)
                    for size_tag in ['sz','szCs']:
                        existing=rp.getElementsByTagName('w:'+size_tag)
                        size=existing[0] if existing else element('w',size_tag)
                        size.setAttribute('w:val','21')
                        if not existing:rp.appendChild(size)
                cell.appendChild(para);row.appendChild(cell)
            node.appendChild(row)
        parent.appendChild(node)
    def mr(value):
        run=element('m','r');t=element('m','t')
        t.appendChild(document.createTextNode(value));run.appendChild(t)
        return run
    def sub(base,index):
        node=element('m','sSub');e=element('m','e');e.appendChild(mr(base))
        s=element('m','sub');s.appendChild(mr(index));node.appendChild(e);node.appendChild(s)
        return node
    def fraction(top,bottom):
        node=element('m','f')
        for kind,value in [('num',top),('den',bottom)]:
            part=element('m',kind);part.appendChild(mr(value));node.appendChild(part)
        return node
    def equation(parent,parts):
        para=p('');pr=para.getElementsByTagName('w:pPr')[0]
        for child in list(pr.childNodes):
            if child.nodeName=='w:ind':pr.removeChild(child)
        jc=element('w','jc');jc.setAttribute('w:val','center');pr.appendChild(jc)
        math=element('m','oMathPara');expr=element('m','oMath')
        for value in parts:expr.appendChild(mr(value) if isinstance(value,str) else value)
        math.appendChild(expr);para.appendChild(math);parent.appendChild(para)
    def fmt(key,field='mean'):return f"{metrics[key][field]:.2f}"
    def ci(stats):return f"[{stats['ci95'][0]:.2f}, {stats['ci95'][1]:.2f}]"
    def replace(prefix,value):
        try:
            target=find_prefix(prefix)
        except StopIteration:
            target=find_prefix(value[:16])
        set_text(target,value)

    replace('基于DQN','基于DQN与PPO的Breakout智能体课程报告')
    replace('三 DQN','三 DQN与PPO方法')
    replace('本课程项目实现了','本课程项目实现了基于DQN与PPO的Breakout图像控制系统。两种方法共用physics-v3固定球速环境、四帧图像输入、原始击砖奖励和完整五生命游戏。DQN使用经验回放、目标Q网络与epsilon-greedy；PPO使用策略/价值网络、GAE、概率比裁剪与熵奖励。项目补充独立梯度裁剪、稳健价值回归、逐局固定开发种子、模型选择、自动测试及网页演示。')
    replace('在当前物理版本下',f'原DQN开发最佳均分283.48及30局复测260.27仍作为开发诊断保留。本次另冻结模型，在121000—121049共50个相同种子上测试：DQN均分{fmt("dqn")}，PPO λ=0.98为{fmt("ppo98")}，λ=0.99为{fmt("ppo99")}，随机策略为{fmt("random")}。主PPO候选按开发均分选定，未按测试成绩更换。两种算法具有不同多阶段训练历史，本次比较固定模型，不构成同预算的普遍算法排名。')
    replace('关键词','关键词  DQN  PPO  Breakout  图像强化学习  GAE  模型评估')
    replace('本项目的目标不是','本项目在统一自制环境中实现DQN与PPO两条训练链路，比较价值学习与策略优化的原理、工程实现、训练波动和固定模型测试结果。项目不复现原版Atari基准，也不把实现技巧作为新算法；重点是可运行、可复核的课程实验。')
    replace('围绕这一目标','围绕这一目标，项目完成规则重建与测试、统一图像接口、DQN和PPO训练恢复、开发选模与留出测试，以及网页实时展示。DQN回放与PPO当前rollout分别持久化必要状态；比较结果均保留配置、模型散列和逐局分数。')
    replace('本项目采用经典DQN','本项目实现经典DQN和PPO及若干可配置实现变体，未提出新的强化学习算法。主要工作包括任务重建、统一实验系统、同起点消融、负结果保留与可复核的同种子模型比较。')
    replace('当前最重要的局限', '本次已经完成相同50个种子的Random、DQN与PPO测试，原283.48和260.27仍属于开发分数，不作为最终测试成绩。主要局限转为训练种子数量不足、各算法多阶段训练预算和搜索历史不同，以及尚未验证跨规则泛化。')
    replace('后续工作将按以下顺序', '后续先对入围参数使用至少3个独立训练种子重复，再在未参与调参的新测试集检验。随后分析低分局、边缘反弹、第二面墙与无进展长局，分别消融策略学习率、GAE、回放容量和表征结构；继续报告低分位、截断率和配对区间。')
    replace('本项目完成了从环境重建','本项目完成了环境重建、DQN与PPO训练、开发选模、固定模型测试和网页展示。DQN用回放和目标网络学习动作价值，PPO用新轨迹与裁剪目标更新策略。统一输入和游戏契约使差异可以在相同任务下检验；原理、实现、训练过程和局限均保留证据。')
    replace('当前推荐模型在50局', f'原DQN开发最佳均分为283.48；新50局测试中DQN、PPO λ=0.98和λ=0.99的均分分别为{fmt("dqn")}、{fmt("ppo98")}和{fmt("ppo99")}。两条链路均学会有效控制，但训练和局间波动仍明显。最后一部分进一步比较两种方法的机制、低分风险和配对差，并说明预算不匹配带来的解释边界。')
    for para in document.getElementsByTagName('w:p'):
        existing=''.join(t.firstChild.data if t.firstChild else '' for t in para.getElementsByTagName('w:t'))
        replacements={
            '将像素控制任务转化为DQN可学习问题':'建立DQN与PPO共用的像素控制任务',
            '500k基础训练 200k续训配置 检查点和回放保存':'DQN多阶段训练、PPO同起点对照与状态保存',
            '固定开发种子 均值和中位数选模 逐局记录':'开发选模、同种子50局测试和逐局证据',
        }
        if existing in replacements:set_text(para,replacements[existing])

    theory=block('theory',find_prefix('四 系统实现'))
    text(theory,'PPO策略与价值学习',2)
    text(theory,'PPO直接学习四动作分布πθ(a|s)，以状态价值Vφ(s)作为基线。训练采样动作，确定性评估取最大动作概率。它按旧策略收集当前rollout，再做多轮小批量更新，随后重新采样；不使用DQN的长期回放、目标Q网络或epsilon-greedy。')
    text(theory,'GAE优势估计',2)
    equation(theory,[sub('δ','t'),'ᴠ = ',sub('r','t'),' + γ(1 − ',sub('d','t'),')Vφ,old(',sub('s','t+1'),') − Vφ,old(',sub('s','t'),')'])
    equation(theory,[sub('Â','t'),' = ∑ₗ₌₀ᴷ⁻¹(γλ)ˡ',sub('δ','t+l'),'ᴠ，  ',sub('R̂','t'),' = ',sub('Â','t'),' + Vφ,old(',sub('s','t'),')'])
    text(theory,'求和到真正终止或rollout边界；时间截断使用实际终止观测自举。GAE目标保持固定，小批量优势标准化不改变评估得分。γ=0.995时，λ从0.98变为0.99使权重衰减尺度约从40增至67步，也可能增大方差；这不是硬截断。')
    text(theory,'概率比裁剪和总损失',2)
    equation(theory,[sub('ρ','t'),'(θ) = ',fraction('πθ(aₜ | sₜ)','πθ,old(aₜ | sₜ)')])
    equation(theory,['Lᶜˡⁱᵖ(θ) = E[min(ρₜÂₜ, clip(ρₜ, 1−ε, 1+ε)Âₜ)]'])
    equation(theory,['ℒ = −Lᶜˡⁱᵖ + cᵥ E[ℓᵥ(R̂−Vφ(s))] − cₕ E[H(πθ)]'])
    text(theory,'ε控制概率比裁剪范围，价值损失拟合固定returns，熵奖励保留探索。裁剪不保证每次更新提分，项目还保留近似KL保护。独立梯度裁剪与稳健价值回归属于实现变体，不能替代同条件实验验证。')

    implementation=block('implementation',find_prefix('五 实验设置'))
    text(implementation,'PPO双网络与训练管理',2)
    text(implementation,'标准PPO共享NatureCNN，改进版使用独立策略/价值CNN：每个分支提取512维特征，策略头输出4个logits，价值头输出1个标量。迁移复制原权重，初始动作概率与价值预测不变；参数组改变后重置Adam，匹配对照也重置。推理只使用策略分支。')
    text(implementation,'Balanced PPO对策略、价值梯度分别按0.5和1.0裁剪，价值损失为2×Huber(delta=10)，小误差区与MSE相同，大误差区减弱异常残差影响。DualRate的额外价值拟合保留原returns、advantages和旧概率，策略梯度设为None，冻结策略权重及其Adam状态。')
    table(implementation,[['模块','PPO职责'],['train_ppo.py','轨迹采样、退火、保存和自动测试'],['balanced_ppo.py / dual_rate_ppo.py','独立裁剪、Huber、双学习率消融'],['episode_seeds.py / callbacks.py','逐局固定种子、开发诊断和心跳'],['checkpoint_selection.py / model_io.py','带均分约束的选模与模型变体加载']])
    last=p('DQN与PPO共用游戏和4×84×84像素输入。PPO续训重新采集rollout，DQN则需要检查回放是否匹配；展示时分别选最大动作概率或最大Q值。辅助奖励仅用于单独消融，所有最终测试均为原生分数。')
    implementation.appendChild(last)
    insert_image(last,'ppo_actor_critic.png','图1001  PPO独立策略与价值网络：共同输入，分别裁剪梯度','rIdPPOReportArchitecture',201)

    settings=block('settings',find_prefix('六 实验结果'))
    text(settings,'PPO训练配置与数据隔离',2)
    table(settings,[['参数','两组共同设置'],['新增预算 / 实际步数','500000 / 503808'],['环境数×rollout / batch / epoch','16×256=4096 / 256 / 4'],['学习率','2e−5 → 5e−6'],['γ / λ','0.995 / 0.98或0.99（唯一训练差异）'],['clip / KL目标','0.06 → 0.03 / 0.0075'],['熵系数 / 价值权重','0.0003 → 0.0001 / 0.25'],['梯度上限 / 训练种子','策略0.5、价值1.0 / 47']])
    text(settings,'源选择使用101000—101031；开发使用111000—111031，每100k评估32局；最终测试为121000—121049共50局。开发环境按局固定seed，不受前局RNG消耗影响。两组从同一验证选定的Balanced检查点开始，均重置Adam，执行完整预算。')
    last=p('最终保留开发均分最佳；中位数最佳仅在均分达到前者95%且中位数更高时替换。最后一轮两个稳定候选都未通过均分约束，实际选中均分最佳。完整更新后KL与小批量KL汇总区分记录；评估每约5秒或每完成一局更新心跳。')
    settings.appendChild(last)
    insert_image(last,'ppo_training_curves.png','图1002  PPO同起点GAE对照：训练、开发评估与更新后的KL','rIdPPOReportTraining',202)

    results=block('results',find_prefix('七 结果分析'))
    text(results,'PPO迭代结果与最终独立测试',2)
    text(results,'首轮标准PPO在20局测试均分35.00；稳定续训技巧的新30局为43.33，未超过仅续训的53.00。长回报原奖励、辅助奖励在另一组50局分别为88.54和57.34；Balanced后续50局为100.76，对照81.84，但差值区间跨零。这些来自不同测试集，不直接串成提升曲线。')
    text(results,'价值学习率3倍且额外拟合2遍的组合，在97条更新中有96.9%降低训练MSE，相对降幅中位数约19%，但最终按开发均分仍选中起点，测试128.12与起点相同，对照134.00。更低训练误差不等于更高策略得分。300k稳定模型开发均分只比起点低0.0625，中位数却从80.5升至132.5，已在新验证集复核。')
    table(results,[['同种子50局','均分','中位数','25%分位','截断'],*[ [metrics[k]['label'],fmt(k),fmt(k,'median'),fmt(k,'p25'),f"{metrics[k]['truncated']}/50"] for k in ['source','ppo98','ppo99'] ]])
    stats=pair['ppo99_minus_ppo98']
    text(results,f'最后一轮λ=0.99减0.98的配对均分差为{stats["mean"]:.2f}，95%回合bootstrap区间{ci(stats)}。两组各只有一个训练种子，应结合低分风险和截断分析，不能仅凭样本均分较高推断普遍优势。')

    final=block('comparison')
    text(final,'十一 DQN与PPO方法比较',1)
    text(final,'学习机制与工程取舍',2)
    table(final,[['方面','DQN','PPO'],['优化对象','四动作Q值的TD回归','裁剪策略目标与状态价值'],['数据利用','离策略，长期回放复用','同策略，当前rollout内多轮更新'],['稳定机制','目标网络、回放、Huber','概率比裁剪、优势标准化、KL'],['探索','epsilon-greedy','动作分布采样与熵奖励'],['网络和推理','在线Q网络及目标网络','独立策略/价值CNN，推理仅策略'],['恢复','模型和回放分离保存','模型与优化器保存，重新采样']])
    text(final,'两者均需要独立开发选模；DQN回放复用历史经验但占用图像内存，PPO限制旧轨迹复用而及时反映当前策略。损失目标和尺度不同，不直接比较损失绝对值，也不能说裁剪或回放保证了训练稳定。')
    text(final,'统一测试协议与定量比较',2)
    text(final,'冻结DQN开发流程推荐模型及两种PPO设置，在同一physics-v3、原始奖励、完整五生命游戏、4×84×84输入和121000—121049种子上各测试50局。均为CPU确定性动作，随机策略均匀采样。主PPO按开发均分196.00高于171.09375选定λ=0.98，λ=0.99仅作补充，不依据测试均分更换主候选。')
    table(final,[['固定模型','均分','标准差','中位数','<50分','截断'],*[ [metrics[k]['label'],fmt(k),fmt(k,'std'),fmt(k,'median'),f"{metrics[k]['below50']}/50",f"{metrics[k]['truncated']}/50"] for k in ['random','dqn','ppo98','ppo99'] ]])
    stats=pair['dqn_minus_ppo98']
    last=p(f'DQN减PPO λ=0.98的配对均分差为{stats["mean"]:.2f}，胜/平/负为{stats["wins"]}/{stats["ties"]}/{stats["losses"]}。按完整种子对重采样20000次，随机数种子20261022，95%区间{ci(stats)}。区间描述固定模型在当前样本上的差异，不覆盖重复训练波动。')
    final.appendChild(last)
    insert_image(last,'dqn_ppo_comparison.png','图1003  DQN与PPO在相同50个测试种子上的得分分布与配对差','rIdPPOReportComparison',203)
    text(final,'比较结论与适用范围',2)
    text(final,f'相同50局的DQN均分为{fmt("dqn")}，PPO主候选为{fmt("ppo98")}，补充GAE设置为{fmt("ppo99")}。应结合中位数、低分位、截断和逐局配对差评价，而非最高分。时间截断并非通关，局长增加也不必然意味着单位交互得分效率提高。')
    text(final,'本次比较不是同预算算法排名：DQN与PPO具有不同的多阶段训练起点、累计交互和超参数搜索。结果仅支持这些固定模型在当前任务上的比较。后续需统一从头预算、多训练种子及未参与调参的新测试集；本报告的贡献是两种方法在同一环境中的可运行实现、完整证据及工程取舍。')

    for name in ['docProps/core.xml']:
        if name in files:
            files[name]=files[name].replace('基于DQN的Breakout'.encode(),'基于DQN与PPO的Breakout'.encode())
