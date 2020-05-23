#-*- coding:utf-8 -*-
import os,sys
if not os.path.exists('/home/aistudio/pypi'):
	os.mkdir('/home/aistudio/pypi')
if not os.path.exists('/home/aistudio/pypi2'):
    os.mkdir('/home/aistudio/pypi2')
print(
'''
     █████╗ ██╗    
    ██╔══██╗██║    python install.py   ----->自动模式安装tensorflow
    ███████║██║    python install.py 0 ----->自动模式安装tensorflow 强制下载模式
    ██╔══██║██║    python install.py 1 ----->自动模式安装torch
    ██║  ██║██║    python install.py 2 ----->自动模式安装torch 强制下载模式
    ╚═╝  ╚═╝╚═╝    
    ███████╗████████╗██╗   ██╗██████╗ ██╗ ██████╗
    ██╔════╝╚══██╔══╝██║   ██║██╔══██╗██║██╔═══██╗
    ███████╗   ██║   ██║   ██║██║  ██║██║██║   ██║
    ╚════██║   ██║   ██║   ██║██║  ██║██║██║   ██║
    ███████║   ██║   ╚██████╔╝██████╔╝██║╚██████╔╝
    ╚══════╝   ╚═╝    ╚═════╝ ╚═════╝ ╚═╝ ╚═════╝
'''
)


cmd=[
    'wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/cudatoolkit-9.0-h13b8566_0.conda -P /home/aistudio/pypi2/',
    'python3.6 -m pip download tensorflow-gpu==1.12.0 keras==2.1.5 -i https://pypi.tuna.tsinghua.edu.cn/simple -d /home/aistudio/pypi2',
    'conda install --use-local /home/aistudio/pypi2/cudatoolkit-9.0-h13b8566_0.conda -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/',
    'python3.6 -m pip install tensorflow-gpu==1.12.0 keras==2.1.5 --no-index -f /home/aistudio/pypi2']

cmd2=['pip download bs4 lxml xlwt xlrd seaborn -i https://pypi.tuna.tsinghua.edu.cn/simple -d /home/aistudio/pypi',
      'pip download torch==1.4_实体识别_1_3的O.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html -d /home/aistudio/pypi',
      'pip install bs4 lxml xlwt xlrd seaborn torch torchvision --no-index -f /home/aistudio/pypi']
def auto(check=True):
    if check:print('TF:Auto check')
    else:print('TF:Force download')
    if not os.path.exists('/home/aistudio/pypi2/cudatoolkit-9.0-h13b8566_0.conda'):os.system(cmd[0])
    if check:
        if not os.path.exists('/home/aistudio/pypi2/tensorflow_gpu-1.12.0-cp36-cp36m-manylinux1_x86_64.whl'):os.system(cmd[1])
    else:os.system(cmd[1])
    if not 'cudatoolkit' in os.popen('conda list cudatoolkit').read():os.system(cmd[2])
    if os.popen('python3.6 -m pip show keras').read()=='':os.system(cmd[3])
    print('TF已就绪于python3.6,请在命令行输入 python3.6 进行测试')
def auto2(check=True):
    if check:
        print('Torch:auto check')
        if not 'lxml' in ' '.join(os.listdir('/home/aistudio/pypi/')):
                os.system(cmd2[0])
        if not 'torch' in ' '.join(os.listdir('/home/aistudio/pypi/')):
                os.system(cmd2[1])
    else:
        print('Torch:force download')
        os.system(cmd2[0]);os.system(cmd2[1])
    if os.popen('pip show torch').read()=='' and 'torch' in ' '.join(os.listdir('/home/aistudio/pypi/')):
            os.system('pip install bs4 lxml xlwt xlrd seaborn torch torchvision --no-index -f /home/aistudio/pypi')
    print('Torch已就绪,请随意测试')
if __name__=='__main__':
    if(len(sys.argv)<2):
        auto()
    elif sys.argv[1] in ['0','1','2']:
        if sys.argv[1]=='0':
            auto(False)
        if sys.argv[1]=='1':
            auto2()
        if sys.argv[1]=='2':
            auto2(False)


