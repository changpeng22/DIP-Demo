function varargout = app(varargin)
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @app_OpeningFcn, ...
                   'gui_OutputFcn',  @app_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

% --- Executes on button press in selpath_button.
function selpath_button_Callback(hObject, eventdata, handles)
%获取图像
global img;
[Fnameh,Pnameh]=uigetfile({'*.png';'*.jpg';'*.jpeg';'*.*'});
img_path=[Pnameh,Fnameh];set(handles.selpath_edit,'String',img_path); %设置文本框内容
img= imread(img_path);  
handles.img=img;guidata(hObject,handles); 
axes(handles.axes1);             
imshow(img);title('原始图像'); 

% --- Executes on button press in rotate_button.
function rotate_button_Callback(hObject, eventdata, handles)
%图像旋转
img=handles.img;
rotate_angle=str2double(get(handles.rotate_edit,'String'));%获取rotate_edit数据并转换
rotate_img=imrotate(img,rotate_angle,"bilinear");%二阶线性插值
axes(handles.axes2);           %选择axes2
imshow(rotate_img);title(['旋转图像(',num2str(rotate_angle),')']);  

% --- Executes on button press in resize_button.
function resize_button_Callback(hObject, eventdata, handles)
% 图像大小变换(倍数)
close(figure(1)); %将之前打开的窗口关闭
f1=figure(1);
set(f1,'name','图像大小变换','Numbertitle','off');
img=handles.img;
resize_scale=str2double(get(handles.resize_edit,'String'));%获取变换的倍数
resize_img=imresize(img,resize_scale,'bilinear');%二阶线性插值
imshow(resize_img);
title(['图像变换(',num2str(resize_scale),'倍)']); 


% --- Executes on button press in gray_button.
function gray_button_Callback(hObject, eventdata, handles)
% 图像灰度化处理
img=handles.img;
gray= rgb2gray(img);   %灰度化
axes(handles.axes2);% 选择axes2显示
imshow(gray);
title('灰度图像');

function mean_slider_Callback(hObject, eventdata, handles)
% 高斯噪声均值
img=handles.img;
ns_mean=get(hObject,'Value');%高斯噪声均值-1~1
handles.ns_mean=ns_mean;guidata(hObject,handles);
ns_var=handles.ns_var; % 高斯噪声方差0-4
gauss_img=imnoise(img,'gaussian',ns_mean,ns_var);%对图片加入噪声
handles.noise_img=gauss_img;guidata(hObject,handles); % 对加噪的图像均命名为noise_img，便于参数传递
axes(handles.axes2);
imshow(gauss_img);
title(['高斯噪声(mean=',num2str(ns_mean),' ','var=',num2str(ns_var),')']);

% --- Executes on slider movement.
function var_slider_Callback(hObject, eventdata, handles)
% 高斯噪声方差
img=handles.img;
ns_var=get(hObject,'Value');%高斯噪声方差0-4
handles.ns_var=ns_var;guidata(hObject,handles);
ns_mean=handles.ns_mean;
gauss_img=imnoise(img,'gaussian',ns_mean,ns_var);%对图片加入噪声
handles.noise_img=gauss_img;guidata(hObject,handles); 
axes(handles.axes2);
imshow(gauss_img);
title(['高斯噪声(mean=',num2str(ns_mean),' ','var=',num2str(ns_var),')']);

% --- Executes on slider movement.
function std_var_slider_Callback(hObject, eventdata, handles)
% 高斯去噪


% --- Executes on slider movement.
function noise_density_slider_Callback(hObject, eventdata, handles)
img=handles.img;
ns_density=get(hObject,'Value');%椒盐噪声密度,范围0-0.8
pepper_img=imnoise(img,'salt & pepper',ns_density);%对图片加入噪声
handles.noise_img=pepper_img;guidata(hObject,handles); 
axes(handles.axes2);
imshow(pepper_img);
title(['椒盐噪声(ns-density=',num2str(ns_density),')']);

% --- Executes on slider movement.
function conpress_slider_Callback(hObject, eventdata, handles)
img=handles.img; %读取待压缩的图像
DCT_ratio=get(hObject,'Value');%DCT压缩系数0-0.02
disp(DCT_ratio);
wtbar=waitbar(1.0,[num2str(DCT_ratio),'DCT压缩比图像压缩中...']); %设置提示进度条
tic;
cr= DCT_ratio;
[mm,nm,dim]=size(img);
i1 = im2double(img);
%对图像进行哈达玛变换
t = dctmtx(8);%生成一个8*8 DCT变换矩阵
for k=1:dim
    dctcoe = blkproc(i1(:,:,k),[8 8],'P1*x*P2',t,t');%将图像分割为8*8的子图像进行FFT
    %x就是每一个分成的8*8大小的块，P1*x*P2相当于像素块的处理函数，p1 = T p2 = T’,
    %也就是fun = p1*x*p2' = T*x*T'的功能是进行离散余弦变换
    coevar = im2col(dctcoe,[8 8],'distinct');%降变换系数矩阵重新排列
    coe = coevar;
    [y,ind] = sort(coevar);
    [m,n] = size(coevar);%根据压缩比确定要变0的系数个数
    %舍去不重要的系数
    snum = 64 - 64 * cr;
    for img = 1:n
        coe(ind(1:floor(snum)),img) = 0;%将最小的snum个变换系数清0
    end
    b2 = col2im(coe,[8 8],[mm nm],'distinct');%重新排列系数矩阵
    %对截取后的变换系数进行哈达玛逆变换
    i2 = blkproc(b2,[8 8],'P1*x*P2',t',t);%对截取后的变换系数进行哈达玛逆变换
    [m1,n1]=size(i2);
    conpress_img(:,:,k)=i2;
end
axes(handles.axes2);
imshow(conpress_img);title([num2str(DCT_ratio),'DCT压缩比图像']);
imwrite(conpress_img,'C:\MATLAB\bin\DSP\DSP_demo/Images/conpress.png');
close(wtbar);% 关闭提示进度条
toc;


% 图像卷积(卷积核)+滤波器
function popmenu_Callback(hObject, eventdata, handles)
% contents = cellstr(get(hObject,'String')); % cell结构体数组
%contents{get(hObject,'Value')} %获取popmenu中的选中值
filter_num= get(handles.popmenu, 'Value'); % 获取popmenu中的第i个选项编号
switch(filter_num)
    case 1 
        filter = 1/9*ones(3,3);
    case 2
        filter = 1/10*[1 1 1; 1 2 1; 1 1 1];
    case 3
        filter = 1/16*[1 2 1; 2 4 2; 1 2 1];
    case 4 
        filter = [0 -1 0; -1 5 -1; 0 -1 0];
    case 5
        filter = [-1 -1 -1; -1 9 -1; -1 -1 -1];
    case 6
        filter = [1 -2 1; -2 5 -2; 1 -2 1];
    case 7 
        filter = [0 0 0; -1 1 0; 0 0 0];
    case 8
        filter = [0 -1 0; 0 1 0; 0 0 0];
    case 9
        filter = [-1 0 0; 0 1 0; 0 0 0];
    case 10
        filter = [-1 -1 -1 -1 -1; 0 0 0 0 0; 1 1 1 1 1];
    case 11
        filter = [-1 0 1; -1 0 1; -1 0 1; -1 0 1; -1 0 1];
    case 12 
        filter = [-1 0 -1; 0 4  0; -1 0 -1];
    case 13
        filter = [-1 -1 -1; -1 8 -1; -1 -1 -1];
    case 14
        filter = [-1 -1 -1; -1 9 -1; -1 -1 -1];
    case 15
        filter = [1 -2 1; -2 4 -2;1 -2 1];
    case 16 
        filter = [1 1 1;1 -2 1;-1 -1 -1];
    case 17
        filter = [1 1 1; -1 -2 1; -1 -1 1];
    case 18
        filter = [-1 1 1; -1 -2 1; -1 1 1];
    case 19
        filter = [-1 -1 1; -1 -2 1; 1 1 1];
    case 20
        filter = [-1 -1 -1; 1 -2 1; 1 1 1];
    case 21
        filter = [1 -1 -1; 1 -2 -1; 1 1 1];
    case 22
        filter = [1 1 -1; 1 -2 -1; 1 1 -1];
    case 23
        filter = [1 1 1; 1 -2 -1; 1 -1 -1];
    case 24
        filter=[];
end
set(handles.filter_table,'Data',filter);
% 图像卷积，卷积核为二维的filter
img=handles.noise_img; % noise_img为高斯噪声或椒盐噪声
[~,~,dim]=size(img);% 获取图像的维度dim
for i=1:dim
    img(:,:,i)=filter2(filter,img(:,:,i)); % 每个图像通道进行卷积
    img_conv=img;
end
axes(handles.axes2);% 选择axes2显示
imshow(img_conv);
contents = cellstr(get(hObject,'String')); %获取所有的选择项
title(contents{get(hObject,'Value')});


% --- Executes on button press in screenshot_button.
function screenshot_button_Callback(hObject, eventdata, handles)
% 选定矩形区域自动截图
img=handles.img;
axes(handles.axes2);
%title('划定截图，右键截图');
imshow(img);
t=imrect; %绘制需要截图的矩形区域
pos=getPosition(t);
[screenshot,rect] = imcrop(img,pos);
imshow(screenshot);title('截图');


% --- Executes on button press in mosaic_button.
function mosaic_button_Callback(hObject, eventdata, handles)
%选定图像某一矩形区域打马赛克
img = handles.img;
[~,~,dim]=size(img);
axes(handles.axes2);
imshow(img); %原始图片显示
pos=getPosition(imrect);% 获取截图区域
a=int16(pos(1,1)); %a,b,c,d分别为截图区域的四个角向量
b=int16(pos(1,2));
c=int16(pos(1,3));
d=int16(pos(1,4));
rec_img=imcrop(img,pos); % 对选择的区域截取
[h w,~] = size(rec_img);
imgn = rec_img; % 定义截取区域相同大小的空间
%设置马赛克区域n*n像素块大小
n = min(floor(pos));
nh = floor(h/n)*n;%将不一定是n的整数倍的图像大小化为整数倍
nw = floor(w/n)*n;
%对rgb三个通道进行循环处理
for k = 1:dim
    for j = 1:n:nh
        for i = 1:n:nw
            imgn(j:j+n-1, i:i+n-1, k) =mean(mean(rec_img(j:j+n-1, i:i+n-1, k)));%对列进行取均值处理
        end
        imgn(j:j+n-1,nw:w , k ) = mean(mean(rec_img(j:j+n-1,nw:w , k)));%处理最后的列
    end
    for i = 1:n:nw
        imgn(nh:h, i:i+n-1, k) = mean(mean(rec_img(nh:h, i:i+n-1, k)));%处理最后的行
    end
    imgn(nh:h, nw:w, k) = mean(mean(rec_img(nh:h, nw:w , k)));%处理最后的角
end
[h1,w1,~]=size(imgn);
a1=length(b:b+d);b1=length(a:a+c);
a2=b:b+d+(h1-a1); % 处理矩阵的行不匹配
b2=a:a+c+(w1-b1); %处理矩阵的列不匹配
img(a2,b2,1:dim)=imgn(:,:,1:dim); 
imshow(img);%显示打马赛克后的图片
title('图像打马赛克后');

% --- Executes on selection change in flip_popmenu.
function flip_popmenu_Callback(hObject, eventdata, handles)
%图像翻转
img=handles.img;
flip_num= get(handles.flip_popmenu, 'Value'); % 获取flip_popmenu中的第i个选项编号
axes(handles.axes2);
switch(flip_num)
    case 1
        h_img=flip(img,2); %水平翻转
        imshow(h_img);
        title('水平翻转');
    case 2 
        v_img=flip(img,1); % 竖直翻转
        imshow(v_img);
        title('竖直翻转');
end

% --- Executes on selection change in sp_effects_popmenu.
function sp_effects_popmenu_Callback(hObject, eventdata, handles)
%特效:1.运动模糊,
img=handles.img;
spef_num=get(handles.sp_effects_popmenu,'Value');
switch(spef_num)
    case 1 %运动模糊
        h_vague=fspecial('motion',20,20);
        spef_img=imfilter(img,h_vague,'conv','circular');
    case 2 % 交互式提取ROI
        close(figure(1)); %将之前打开的figure1窗口关闭
        f1=figure(1); %绘制所需提取的区域
        set(f1,'name','ROI提取:鼠标左键选择目标边缘，右键或enter结束','Numbertitle','off');
        imshow(img, 'Border','tight'); hold on;
        P = []; num = 1;
        [x, y, k] = ginput(1);
        P = [P; x y];
        plot(x, y, 'r.', 'MarkerSize', 15);
        while k == 1
            num = num + 1;
            [x, y, k] = ginput(1);
            if k ~= 1
            break;
        end
        P = [P; x y];
        plot(x, y, 'r.', 'MarkerSize', 15);
        if num > 1
            line([P(num-1, 1) P(num, 1)], [P(num-1, 2) P(num, 2)], 'Color', 'r', 'LineWidth', 1.2);
        end
        end
        line([P(1, 1) P(end, 1)], [P(1, 2) P(end, 2)], 'Color', 'r', 'LineWidth', 2);
        mf = getframe(gca);
        If = frame2im(mf);
        img = If;
        hsv = rgb2hsv(img);
        s = hsv(:, :, 2);
        bw = im2bw(s, graythresh(s));
        bw_fill = imfill(bw, 'holes');
        spef_img=img;
        if ~isequal(bw_fill, [size(spef_img,1), size(spef_img,2)])
            bw_fill = imresize(bw_fill, [size(spef_img,1), size(spef_img,2)], 'bilinear');
        end
        for i = 1:3
            spef_img(:, :, i) = spef_img(:, :, i).*uint8(bw_fill);
        end
        close(figure(1)); %关闭figure1窗口
    case 3
        % 反色
        img=handles.img;
        [m,n,dim]=size(img);
        for i=1:dim
            img(:,:,i)=(255-img(:,:,i));
        end
        axes(handles.axes2);
        imshow(img);title('图像反色');
        
end
axes(handles.axes2);
imshow(spef_img);
contents = cellstr(get(hObject,'String')); %获取所有的选择项
title(['特效:',contents{get(hObject,'Value')}]);


% --- Executes on button press in watermark_button.
function watermark_button_Callback(hObject, eventdata, handles)
% 水印按钮
close(figure(1)); %将之前打开的窗口关闭
f1=figure(1); %绘制所需提取的区域
set(f1,'name','水印处理','Numbertitle','off');

set(gca,'xtick',[],'ytick',[]);
axis off; %不显示坐标轴
word=get(handles.watermark_edit,'String');
text(0,0,word,'color','black','FontSize',50','FontWeight','bold','Units','normalized','HorizontalAlignment','center','Position',[0.25,0.75]);
saveas(gcf,'tmp','jpg');

%嵌入水印的程序代码
M=512; %原图像长度
N=64; %水印图像长度
K=8;  
I=zeros(M,M);J=zeros(M,M);BLOCK = zeros(K,K);
%显示原图像
% [Fnameh,Pnameh]=uigetfile({'*.png';'*.jpg';'*.jpeg';'*.*'});
% img_path=[Pnameh,Fnameh];set(handles.selpath_edit,'String',img_path); 

subplot(3,2,1);I=handles.img;imshow(I);
title('原始公开图像');
%显示水印图像，水印图像最好为二值图像。
subplot(3,2,2);J=imread("tmp.jpg",'jpg');imshow(J);title('水印图像');

%水印嵌入
for p=1:N
    for q=1:N
        x=(p-1)*K+1; y=(q-1)*K+1;
        BLOCK=I(x:x+K-1,y:y+K-1);%将原始图像分割
        BLOCK=dct2(BLOCK);%原始图像进行DCT变换
    if J(p,q)==0%水印图像为二值图像
        a=-1;
    else
        a=1;
    end
    BLOCK(1,1)=BLOCK(1,1)*(1+a*0.03);%a为调制因子，根据水印的值调制a的正负
    BLOCK=idct2(BLOCK);%做IDCT反变换
    I(x:x+K-1,y:y+K-1)=BLOCK;
    end
end

%显示嵌入水印后的图像
subplot(3,2,3);imshow(I);title('嵌入水印后的图像');

%从嵌入水印的图像中提取水印
I=imread('23.JPG');%嵌入水印的图像
J=imread('tmp.jpg','jpg');
I=imnoise(I,'gaussian',0,0.01);
subplot(3,2,4);imshow(I,[]);%图像加上高斯噪声攻击
title('加入高斯噪声');
for p=1:N
    for q=1:N
        x=(p-1)*K+1;
        y=(q-1)*K+1;
        BLOCK1=I(x:x+K-1,y:y+K-1);%分割图像
        BLOCK2=J(x:x+K-1,y:y+K-1);%分割图像
        BLOCK1=dct2(BLOCK1);
        BLOCK2=dct2(BLOCK2);%DCT变换
        a=BLOCK2(1,1)/BLOCK1(1,1)-1;%两图像在各个分割块对比
    if a<0
        W(p,q)=0;
    else
        W(p,q)=1;
    end
    end
end
%显示提取的水印
subplot(3,2,5);
imshow(W);
title('从含水印图像中提取的水印');


% --- Executes just before app is made visible.
function app_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
guidata(hObject, handles);

% --- Outputs from this function are returned to the command line.
function varargout = app_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;

% 图像旋转角度输入
function rotate_edit_Callback(hObject, eventdata, handles)
function rotate_edit_CreateFcn(hObject, eventdata, handles) 
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% 图像路径输入
function selpath_edit_Callback(hObject, eventdata, handles)   
function selpath_edit_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% 图像变换大小输入
function resize_edit_Callback(hObject, eventdata, handles)
function resize_edit_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
function flip_popmenu_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
% --- Executes during object creation, after setting all properties.
function mean_slider_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
% --- Executes during object creation, after setting all properties.
function var_slider_CreateFcn(hObject, eventdata, handles)
% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
% --- Executes during object creation, after setting all properties.
function noise_density_slider_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
% --- Executes during object creation, after setting all properties.
function axes2_CreateFcn(hObject, eventdata, handles)

function filter_table_CellSelectionCallback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function sp_effects_popmenu_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
function popmenu_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
% --- Executes on mouse press over axes background.
function axes1_ButtonDownFcn(hObject, eventdata, handles)
% --- Executes during object creation, after setting all properties.
function axes1_CreateFcn(hObject, eventdata, handles)
% 添加水印文字
function watermark_edit_Callback(hObject, eventdata, handles)
function watermark_edit_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

%高斯去噪标准差滑块
function std_var_slider_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
% DCT压缩滑块
function conpress_slider_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
