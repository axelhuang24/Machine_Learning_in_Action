from numpy import *

from tkinter import * #一个用于GUI的框架
import regTrees

import matplotlib # 导入matplotlib
matplotlib.use('TkAgg') # 设定后端为 TkAgg
# 以下两个import 声明将 TkAgg 和 Matplotlib图连接起来
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def reDraw(tolS,tolN):
    reDraw.f.clf()        # clear the figure
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get(): # 如果选择了 Model Tree 的话, 是模型树
        if tolN < 2: tolN = 2
        myTree=regTrees.createTree(reDraw.rawDat, regTrees.modelLeaf,\
                                   regTrees.modelErr, (tolS,tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat, \
                                       regTrees.modelTreeEval)
    else: # 否则的话是回归树
        myTree=regTrees.createTree(reDraw.rawDat, ops=(tolS,tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat)
	# 画训练点, 离散的
    reDraw.a.scatter(reDraw.rawDat[:,0].tolist(), reDraw.rawDat[:,1].tolist(), s=5) #use scatter for data set
	# 画测试点, 连续的
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0) #use plot for yHat
    reDraw.canvas.show() # canvas是画布
    
def getInputs():
    try: tolN = int(tolNentry.get())
    except: # 如果输入的不是整型数，删除这个输入并将值设为10
        tolN = 10 
        print("enter Integer for tolN")
        tolNentry.delete(0, END)
        tolNentry.insert(0,'10')
    try: tolS = float(tolSentry.get())
    except: # 如果输入的不是浮点数，删除这个输入并将值设为1.0
        tolS = 1.0 
        print("enter Float for tolS")
        tolSentry.delete(0, END)
        tolSentry.insert(0,'1.0')
    return tolN,tolS

def drawNewTree():
    tolN,tolS = getInputs()#get values from Entry boxes
    reDraw(tolS,tolN)
    
root=Tk()

reDraw.f = Figure(figsize=(5,4), dpi=100) #create canvas
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

Label(root, text="tolN").grid(row=1, column=0)
tolNentry = Entry(root)
tolNentry.grid(row=1, column=1)
tolNentry.insert(0,'10')
Label(root, text="tolS").grid(row=2, column=0)
tolSentry = Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0,'1.0')
Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2, rowspan=3)
chkBtnVar = IntVar()
chkBtn = Checkbutton(root, text="Model Tree", variable = chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)

reDraw.rawDat = mat(regTrees.loadDataSet('sine.txt'))
reDraw.testDat = arange(min(reDraw.rawDat[:,0]),max(reDraw.rawDat[:,0]),0.01)
reDraw(1.0, 10)
               
root.mainloop()