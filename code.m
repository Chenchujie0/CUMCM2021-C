clc,clear;
Ding=xlsread('附件1 近5年402家供应商的相关数据','企业的订货量（m³）','C2:IH403');
Gong=xlsread('附件1 近5年402家供应商的相关数据','供应商的供货量（m³）','C2:IH403');

%% Problem1
    %% 检验
    sum1=0; sum2=0;
    for i=1:402
        for j=1:240
            if Ding(i,j)==0 && Gong(i,j)==0
                sum2=sum2+1;
            end
            if Ding(i,j)==0
                sum1=sum1+1;
            end
        end
    end

    %% 计算供应差值
    Sum_Ding=sum(Ding,2);
    Sum_Gong=sum(Gong,2);
    Gong_ChaZhi=Sum_Gong-Sum_Ding; %供应差值

    %% 计算供应稳定性
    ChaZhi=Gong-Ding;
    ChaZhi_fu=[];
    Gong_WenDing=[];
    for i=1:402
        k=1;
        for j=1:240
            if ChaZhi(i,j)<0
                ChaZhi_fu(i,k)=ChaZhi(i,j);
                k=k+1;
            end
        end
    end
    Gong_WenDing=std(ChaZhi_fu,0,2).*std(ChaZhi_fu,0,2); %供应稳定性

    %% 计算供应弹性
    percent_ChaZhi=zeros(402,240);
    decision=zeros(402,240);
    for i=1:402
        for j=1:240
            if Ding(i,j)~=0
                percent_ChaZhi(i,j)=abs(Gong(i,j)-Ding(i,j))/Ding(i,j);
                if percent_ChaZhi(i,j)<0.5
                    dicision(i,j)=1;
                else 
                    dicision(i,j)=0;
                end
            end
        end
    end
    Gong_TanXing=sum(dicision,2); %供应弹性

    %% 计算供货量与订货量差值的平均增长率，即供应积极性
    Ding_month=[]; %每月订货量
    Gong_month=[]; %每月供货量
    for i=1:402 
        temp1=0;
        temp2=0;
        for j=1:240 
            temp1=temp1+Ding(i,j);
            temp2=temp2+Gong(i,j);
            if mod(j,4)==0
                Ding_month(i,j/4)=temp1;
                Gong_month(i,j/4)=temp2;
                temp1=0;
                temp2=0;
            end
        end
    end

    %每月供货量与订货量的差值
    ChaZhi_month=[];
    for i=1:402
        for j=1:60 
            ChaZhi_month(i,j)=Gong_month(i,j)-Ding_month(i,j); 
        end
    end

    %供货量与订货量差值每月的增长率
    Gong_ChaZhi_Growth_month=[];
    for i=1:402
        for j=1:12
            temp=0;
            for k=1:4
                if ChaZhi_month(i,j+12*(k-1))==0
                    continue;
                end
                temp=temp+(ChaZhi_month(i,j+12*k)-ChaZhi_month(i,j+12*(k-1)))/abs(ChaZhi_month(i,j+12*(k-1)));
            end
            Gong_ChaZhi_Growth_month(i,j)=temp/4;
        end
    end
    Gong_ChaZhi_Growth=mean(Gong_ChaZhi_Growth_month,2); %供货量与订货量差值的平均增长率

    %% 计算供应能力
    Ability=[];
    [Ability,index_Gong]=max(Gong,[],2);

    %% 归一化
    NGong_ChaZhi=normalize(Gong_ChaZhi,'range');
    NGong_WenDing=normalize(Gong_WenDing,'range');
    NGong_TanXing=normalize(Gong_TanXing,'range');
    NGong_ChaZhi_Growth=normalize(Gong_ChaZhi_Growth,'range');
    NAbility=normalize(Ability,'range');
    
    %% 构成对比矩阵并验证一致性
    A = [ 1 3 5 7 4;
         1/3 1 2 5 2;
         1/5 1/2 1 2 1/2;
         1/7 1/5 1/2 1 1/3;
         1/4 1/2 2 3 1];
    W = prod(A, 2);               % 计算每一行乘积
    n = size(A, 1);               % 矩阵行数
    W = nthroot(W, n);            % 计算n次方根
    W = W / sum(W);               % 归一化处理, 计算特征向量, 即权向量
    Lmax = mean((A * W) ./ W);    % 计算最大特征值
    n = 5; 
    CI = (Lmax - n) / (n - 1);    % 计算一致性指标
    RI = 1.12; 
    CR = CI / RI;
    if CR<0.1
        str='矩阵一满足一致性要求'
    else
        str='矩阵一不满足一致性要求'
    end

    Gong_Grade=zeros(402,1);
    for i=1:402
        Gong_Grade(i,1)=W(1)*NGong_ChaZhi(i,1)+W(2)*NGong_WenDing(i,1)+W(3)*NGong_TanXing(i,1)+W(4)*NGong_ChaZhi_Growth(i,1)+W(5)*NAbility(i,1);
    end
    Temp_Grade=Gong_Grade;

%% Problem2
    %% 2.1最小供货商数量
    %% 求总完成度
    Temp_Grade=Gong_Grade;
    Sum_Ding=sum(Ding,2);
    Sum_Gong=sum(Gong,2);
    Finishment=zeros(402,1);
    for i=1:402 
        Finishment(i,1)=Sum_Gong(i,1)/Sum_Ding(i,1);
    end
    
    %% 求理想供应量与平均理想供应量
    Gong_LiXiang=zeros(402,1);
    for i=1:402 
        [max_G,index]=max(Gong,[],2);
        Gong_LiXiang(i,1)=Finishment(i,1)*max_G(i,1);
    end
    
    %% 每种材料供应商的平均供货能力
    [Fenpei_NUM,txt] = xlsread('附件1 近5年402家供应商的相关数据','供应商的供货量（m³）');
    Sum_a=0; Num_a=0; Sum_b=0; Num_b=0; Sum_c=0; Num_c=0;
    for i=1:402
        if txt{i+1,2}=='A'
            Sum_a=Sum_a+Gong_LiXiang(i,1);
            Num_a=Num_a+1;
        end
        if txt{i+1,2}=='B'
            Sum_b=Sum_b+Gong_LiXiang(i,1);
            Num_b=Num_b+1;
        end
        if txt{i+1,2}=='C'
            Sum_c=Sum_c+Gong_LiXiang(i,1);
            Num_c=Num_c+1;
        end
    end
    XAver_a=Sum_a/Num_a;
    XAver_b=Sum_b/Num_b; 
    XAver_c=Sum_c/Num_c;
    
    %% 随机选取最小消耗率
    Sunhao=xlsread('附件2 近5年8家转运商的相关数据','B2:IG9');
    Sunhao(find(Sunhao==0))=[];
    random_Sunhao = Sunhao(randi(length(Sunhao(:)),1,floor(0.8*length(Sunhao(:)))));
    Sunhao_min=min(min(Sunhao));
    
    %% 求最小经济成本时的供应商分配情况(蒙特卡洛）
    format compact;
    rand('state',sum(clock)); % 初始化随机数发生器
    p0=-10000;
    m0=[];
    tic % 计时开始
    for i=1:10^6
        m=[randi([0,146],1,1);randi([0,134],1,1);randi([0,122],1,1)];
        [f,g]=mengte(m,XAver_a,XAver_b,XAver_c,Sunhao_min);
        if all(g<=0)
            if p0<f
                m0=m;p0=f; % 记录下当前较好的解
            end
        end
    end
    toc; % 计时结束

    %% 2.2供货模型
    %% 供货商排名
    Gong_index=(1:402)';
    Temp_txt=txt(2:403,2);
    tem=0;
    for i=1:402
        for j=i+1:402
            if Temp_Grade(j,1)>Temp_Grade(i,1)
                tem=Gong_index(j,1);
                Gong_index(j,1)=Gong_index(i,1);
                Gong_index(i,1)=tem;
                tem=Temp_Grade(j,1);
                Temp_Grade(j,1)=Temp_Grade(i,1);
                Temp_Grade(i,1)=tem;
                te=Temp_txt{i,1};
                Temp_txt{i,1}=Temp_txt{j,1};
                Temp_txt{j,1}=te;
            end
        end
    end
    
    %% 计算10个时间段每周供货量/订货量平均值
    Gong_aver=zeros(402,24);
    Ding_aver=zeros(402,24);
    temp1=0; temp2=0;
    n1=0; n2=0;
    for i=1:402
        for j=1:24
            for k=1:10
                if Gong(i,j+24*(k-1))~=0
                    temp1=temp1+Gong(i,j+24*(k-1));
                    n1=n1+1;
                end
                if Ding(i,j+24*(k-1))~=0
                    temp2=temp2+Ding(i,j+24*(k-1));
                    n2=n2+1;
                end
            end
            if n1~=0
                Gong_aver(i,j)=temp1/n1;
                n1=0;
                temp1=0;
            else
                Gong_aver(i,j)=0;
            end
            if n2~=0
                Ding_aver(i,j)=temp2/n2;
                n2=0;
                temp2=0;
            else
                Ding_aver(i,j)=0;
            end
        end
    end
    
    %% 供货商能力权重
    A=[1 2 9;1/2 1 8; 1/9 1/8 1];
    [V,D] = eig(A);
    [Lmax,ind] = max(diag(D));     % 求最大特征值及其位置
    W = V(:,ind) / sum(V(:,ind));   % 最大特征值对应的特征向量做归一化, 即权向量
    n = size(A, 1);  
    CI = (Lmax - n) / (n - 1);     % 计算一致性指标
    RI = [0 0 0.58 0.90 1.12 1.24 1.32 1.41 1.45 1.49 1.51]; 
    CR = CI / RI(n);
    if CR<0.1
        str='矩阵满足一致性要求'
    else
        str='矩阵不满足一致性要求'
    end
    
    %% 供货商供货量
    Abi=zeros(402,24);
    Ability=[];
    [Ability,index_Gong]=max(Gong,[],2);
    for i=1:402
        for j=1:24
            Abi(i,j)=W(1)*Gong_aver(i,j)+W(2)*Ding_aver(i,j)+W(3)*Ability(i,1);
        end
    end
    
    %% 供货商分类
    S_a=zeros(Num_a,24); SNuma=[];
    S_b=zeros(Num_b,24); SNumb=[];
    S_c=zeros(Num_c,24); SNumc=[];
    ka=0; kb=0; kc=0;
    for i=1:402
        if Temp_txt{i,1}=='A'
            ka=ka+1;
            S_a(ka,:)=Abi(Gong_index(i,1),:);
            SNuma(end+1)=Gong_index(i,1);
        end
        if Temp_txt{i,1}=='B'
            kb=kb+1;
            S_b(kb,:)=Abi(Gong_index(i,1),:);
            SNumb(end+1)=Gong_index(i,1);
        end
        if Temp_txt{i,1}=='C'
            kc=kc+1;
            S_c(kc,:)=Abi(Gong_index(i,1),:);
            SNumc(end+1)=Gong_index(i,1);
        end
    end
    SNuma=SNuma'; SNumb=SNumb'; SNumc=SNumc';
    
    %% 随机选取最小消耗率
    Sunhao=xlsread('附件2 近5年8家转运商的相关数据','B2:IG9');
    Sunhao_z=Sunhao;
    Sunhao_z(find(Sunhao_z==0))=[];
    random_Sunhao_z = Sunhao_z(randi(length(Sunhao_z(:)),1,floor(0.8*length(Sunhao_z(:)))));
    Sunhao_min1=min(min(Sunhao_z));
    
    %% lingo软件计算出最佳供货数量
    t0=[12938.00 1547.000 3091.000];
    
    %% 订货方案
    Fenpei=num2cell(zeros(3,24));
    Fenpei_NUM=num2cell(zeros(3,24));
    
    Suma=0; Sumb=0; Sumc=0;
    a=[]; b=[]; c=[];
    an=[];bn=[]; cn=[];
    for i=1:24
        for j=1:Num_a
            if S_a(j,i)>6000
                S_a(j,i)=6000;
            end
            Suma=Suma+S_a(j,i); 
            a(end+1)=S_a(j,i);
            an(end+1)=SNuma(j,1);
            if Suma>2*t0(1)
                break;
            end
        end
        Fenpei{1,i}=a;
        Fenpei_NUM{1,i}=an;
        a=[];
        an=[];
        Suma=0;
        
        for j=1:Num_b
            if S_b(j,i)>6000
                S_b(j,i)=6000;
            end
            Sumb=Sumb+S_b(j,i); 
            b(end+1)=S_b(j,i);
            bn(end+1)=SNumb(j,1);
            if Sumb>2*t0(2)
                break;
            end
        end
        Fenpei{2,i}=b;
        Fenpei_NUM{2,i}=bn;
        b=[];
        bn=[];
        Sumb=0;
        
        for j=1:Num_c
            if S_c(j,i)>6000
                S_c(j,i)=6000;
            end
            Sumc=Sumc+S_c(j,i);
            c(end+1)=S_c(j,i);
            cn(end+1)=SNumc(j,1); 
            if Sumc>2*t0(3)
                break;
            end
        end
        Fenpei{3,i}=c;
        Fenpei_NUM{3,i}=cn;
        cn=[];
        c=[];
        Sumc=0;
    end
    
    %% 2.3转运模型
    %% 计算平均损耗率
    Sunhao_aver=zeros(8,24);
    temp=0;
    n=0;
    for i=1:8
        for j=1:24
            for k=1:10
                if Sunhao(i,j+24*(k-1))~=0
                    temp=temp+Sunhao(i,j+24*(k-1));
                    n=n+1;
                end
            end
            if n~=0
                Sunhao_aver(i,j)=temp/n;
                n=0;
                temp=0;
            else
                 Sunhao_aver(i,j)=0;
            end
        end
    end
    
    %% 计算运输稳定性 
    Week=num2cell(zeros(8,24)); %以24周为一组
    temp=[];
    for i=1:8
        for j=1:24
            for k=1:10
                if Sunhao(i,j+24*(k-1))~=0
                    temp(end+1)=Sunhao(i,j+24*(k-1));
                end
            end
            Week{i,j}=temp;
            temp=[];
        end
    end
    Sunhao_Fangcha=zeros(8,24);
    Sum=0;
    for i=1:8
        for j=1:24
            if length(Week{i,j})~=0
                Sunhao_Fangcha(i,j)=std(Week{i,j})*std(Week{i,j});
            else
                Sunhao_Fangcha(i,j)=0;
            end
        end
    end
    
    %% 剔除损耗率为0的情况
    Sunhao_new=[];
    h=0; l=0;
    for i=1:8
        l=0;
        h=h+1;
        for j=1:240
            if Sunhao(i,j)~=0
                l=l+1;
                Sunhao_new(h,l)=Sunhao(i,j);
            end
        end
    end
    
    %% 计算预期损耗率(灰色预测模型)
    Sunhao_Yuce=zeros(8,240);
    E1=0.9917;
    E2=1.0083;
    sym1=0;
    delta=zeros(8,240);
    rho=zeros(8,240);
    for i=1:8
        x0=Sunhao_new(i,:)';
        n=length(x0);
        lamda=x0(1:n-1)./x0(2:n);
        
        for j=1:n-1
            if lamda(j)<E1 || lamda(i)>E2
                sym1=1;
            end
        end
        
        if sym1==1
            for k=0:0.1:50
                x01=x0;
                x01=x01+k;
                n=length(x01);
                lamda=x01(1:n-1)./x01(2:n);
                for k1=1:n-1
                    sym2=0;
                    if lamda(k1)<E1 || lamda(k1)>E2
                        sym2=1;
                        break;
                    end
                end
                if sym2==0
                    x0=x01;
                    break;
                end
            end
        end
        sym1=0;
        range=minmax(lamda');
        x1=cumsum(x0);
        B=[-0.5*(x1(1:n-1)+x1(2:n)),ones(n-1,1)];
        Y=x0(2:n);
        u=B\Y;
        syms x(t)
        x=dsolve(diff(x)+u(1)*x==u(2),x(0)==x0(1));
        xt=vpa(x,6);
        yuce1=subs(x,t,[0:n-1]);
        yuce1=double(yuce1);
        yuce=[x0(1),diff(yuce1)];
        epsilon=x0'-yuce;
        delta(i,:)=[abs(epsilon./x0') zeros(1,240-length(abs(epsilon./x0')))];
        rho(i,:)=[1-(1-0.5*u(1))/(1+0.5*u(1))*lamda' zeros(1,240-length(1-(1-0.5*u(1))/(1+0.5*u(1))*lamda'))];
        Sunhao_Yuce(i,:)=yuce;
    end
    
    %% 计算完成周数
    Sunhao_w=[240 240 117 102 83 216 240 203]';
    Sunhao_wancheng=Sunhao_w/240;
    
    %% 归一化
    NSunhao_aver=normalize(Sunhao_aver,'range');
    NSunhao_Fangcha=normalize(Sunhao_Fangcha,'range');
    NSunhao_Yuce=normalize(Sunhao_Yuce,'range');
    NSunhao_wancheng=normalize(Sunhao_wancheng,'range');
    
    %% 构成对比矩阵并验证一致性
    A = [ 1 1/2 1/3 1;
         2 1 1/2 2
         3 2 1 3
         1 1/2 1/3 1];
    W = prod(A, 2);               % 计算每一行乘积
    n = size(A, 1);               % 矩阵行数
    W = nthroot(W, n);            % 计算n次方根
    W = W / sum(W);               % 归一化处理, 计算特征向量, 即权向量
    Lmax = mean((A * W) ./ W);    % 计算最大特征值
    n = 4; 
    CI = (Lmax - n) / (n - 1);    % 计算一致性指标
    RI = 0.9; 
    CR = CI / RI;
    if CR<0.1
        str='矩阵二满足一致性要求'
    else
        str='矩阵二不满足一致性要求'
    end
    Yun_Grade=zeros(8,1);
    for i=1:8
        for j=1:24
            Yun_Grade(i,j)=W(1)*NSunhao_aver(i,j)+W(2)*NSunhao_Fangcha(i,j)+W(3)*Sunhao_Yuce(i,j)+W(4)*NSunhao_wancheng(i,1);
        end
    end
    
    %% 转运商排名
    YTemp_Grade=Yun_Grade;
    Yun_pai=[1:8]'*ones(1,24);
    ytemp=0;
    for i=1:24
        for j=1:8
            for k=j+1:8
                if YTemp_Grade(j,i)<YTemp_Grade(k,i)
                    ytemp=YTemp_Grade(j,i);
                    YTemp_Grade(j,i)=YTemp_Grade(k,i);
                    YTemp_Grade(k,i)=ytemp;
                    ytemp=Yun_pai(j,i);
                    Yun_pai(j,i)=Yun_pai(k,i);
                    Yun_pai(k,i)=ytemp;
                end
            end
        end
    end 
    
    %% 转运方案 
    Fenpei_yun=num2cell(zeros(3,24));
    Fenpei_yun_num=num2cell(zeros(3,24));
    FPN=Fenpei_NUM;
    FP=Fenpei;
    for i=1:24
        FP(1,i)=strcat(FP(1,i),FP(2,i),FP(3,i));
        FPN(1,i)=strcat(Fenpei_NUM(1,i),Fenpei_NUM(2,i),Fenpei_NUM(3,i));
    end
    FP=FP(1,:);
    FPN=FPN(1,:);
    
    %导出表二
    %FP=Fenpei;
    %FPN=Fenpei_NUM;
    %for i=1:24
    %    FP1(i,:)=[FP{1,i} zeros(1,402-length(FP{1,i}))];
    %    FPN1(i,:)=[FPN{1,i} zeros(1,402-length(FPN{1,i}))];
    %end
    %写入表二
    %number=xlsread('表二数据','供货商编号','B2:JR25');
    %Gonghuo=xlsread('表二数据','供货量','B2:JR25');
    %F=zeros(402,24);
    %for i=1:24
    %    for j=1:277
    %        if number(i,j)~=0
    %            F(number(i,j),i)=Gonghuo(i,j);
    %        end
    %    end
    %end
    %写入表三
    %     F=zeros(402,24*8);
    %     for i=1:24
    %         for j=1:5
    %             for k=1:length(FPNY{j,i})
    %                 F(FPNY{j,i}(1,k),j+8*(i-1))=FPY{j,i}(1,k);
    %             end
    %         end
    %     end
     
    len=[];
    for i=1:24
        len(end+1)=length(FP{1,i});
    end
    for i=1:24
        Sum=0; a=[]; k=1; an=[];
        for j=1:len(i) 
            if FP{1,i}(1,j)>6000
                FP{1,i}(1,j)=6000;
            end
            Sum=Sum+FP{1,i}(1,j); 
            a(end+1)=FP{1,i}(1,j);
            an(end+1)=FPN{1,i}(1,j);
            if Sum>6000
                Fenpei_yun{k,i}=a(1:end-1);
                Fenpei_yun_num{k,i}=an(1:end-1);
                a=[];
                an=[];
                a(end+1)=FP{1,i}(1,j);
                an(end+1)=FPN{1,i}(1,j);
                Sum=FP{1,i}(1,j);
                k=k+1;
                if k>8
                    break;
                end
            end
        end %for j=1:length(FP{1,i})      
    end %for i=1:24

%% problem3
    %% 订货方案/供货方案 
    beta=[];
    xt3=zeros(24,3);
    for i=1:24
        beta(i,:)=[Sunhao_Yuce(Yun_pai(1,i),i) Sunhao_Yuce(Yun_pai(2,i),i) Sunhao_Yuce(Yun_pai(3,i),i) Sunhao_Yuce(Yun_pai(4,i),i)]/100;
        goal=[0,-38986];%T3_min-T1_max
        w=[1,1];
        a=[(0.000011-1)/0.6,(0.000011-1)/0.66,(0.000011-1)/0.72;-beta(i,3),-beta(i,3),-beta(i,3)];
        b=[-28200;beta(i,1)*6000+beta(i,2)*6000-beta(i,3)*12000];
        lb=[2657;1547;3089];
        ub=[42075;35257;29182];
        fun1=@(T) [(beta(i,1)*6000+beta(i,2)*6000+beta(i,3)*(T(1)+T(2)+T(3)-12000))/(T(1)+T(2)+T(3));T(3)-T(1)];
        [xt3(i,:),fval]=fgoalattain(fun1,[2657 1547 3089],goal,w,a,b,[],[],lb,ub);
    end
    
    %计算betap
    betap=[];
    for i=1:24
        betap(end+1)=(beta(i,1)*6000+beta(i,2)*6000+beta(i,3)*(xt3(i,1)+xt3(i,2)+xt3(i,3)-12000))/(xt3(i,1)+xt3(i,2)+xt3(i,3));
    end
    
    %% 分配供货商(p3)
    Fenpei_3=num2cell(zeros(3,24));
    Fenpei_NUM_3=num2cell(zeros(3,24));
    Suma=0; Sumb=0; Sumc=0;
    a=[]; b=[]; c=[];
    an=[];bn=[]; cn=[];
    for i=1:24
        for j=1:Num_a
            if S_a(j,i)>6000
                S_a(j,i)=6000;
            end
            Suma=Suma+S_a(j,i); 
            a(end+1)=S_a(j,i);
            an(end+1)=SNuma(j,1);
            if Suma>2*xt3(i,1)
                break;
            end
        end
        Fenpei_3{1,i}=a;
        Fenpei_NUM_3{1,i}=an;
        a=[];
        an=[];
        Suma=0;
        
        for j=1:Num_b
            if S_b(j,i)>6000
                S_b(j,i)=6000;
            end
            Sumb=Sumb+S_b(j,i); 
            b(end+1)=S_b(j,i);
            bn(end+1)=SNumb(j,1);
            if Sumb>2*xt3(i,2)
                break;
            end
        end
        Fenpei_3{2,i}=b;
        Fenpei_NUM_3{2,i}=bn;
        b=[];
        bn=[];
        Sumb=0;
        
        for j=1:Num_c
            if S_c(j,i)>6000
                S_c(j,i)=6000;
            end
            Sumc=Sumc+S_c(j,i);
            c(end+1)=S_c(j,i);
            cn(end+1)=SNumc(j,1); 
            if Sumc>2*xt3(i,3)
                break;
            end
        end
        Fenpei_3{3,i}=c;
        Fenpei_NUM_3{3,i}=cn;
        cn=[];
        c=[];
        Sumc=0;
    end
    
    %% 分配转运商(p3)
    Fenpei_yun_3=num2cell(zeros(3,24));
    Fenpei_yun_num_3=num2cell(zeros(3,24));
    FPN_3=Fenpei_NUM_3;
    FP_3=Fenpei_3;

    for i=1:24
        FP_3(1,i)=strcat(FP_3(1,i),FP_3(2,i),FP_3(3,i));
        FPN_3(1,i)=strcat(Fenpei_NUM_3(1,i),Fenpei_NUM_3(2,i),Fenpei_NUM_3(3,i));
    end
    FP_3=FP_3(1,:);
    FPN_3=FPN_3(1,:);
     
    len=[];
    for i=1:24
        len(end+1)=length(FPN_3{1,i});
    end
    for i=1:24
        Sum=0; a=[]; k=1; an=[];
        for j=1:len(i) 
            if FP_3{1,i}(1,j)>6000
                FP_3{1,i}(1,j)=6000;
            end
            Sum=Sum+FP_3{1,i}(1,j); 
            a(end+1)=FP_3{1,i}(1,j);
            an(end+1)=FPN_3{1,i}(1,j);
            if Sum>6000
                Fenpei_yun_3{k,i}=a(1:end-1);
                Fenpei_yun_num_3{k,i}=an(1:end-1);
                a=[];
                an=[];
                a(end+1)=FP_3{1,i}(1,j);
                an(end+1)=FPN_3{1,i}(1,j);
                Sum=FP_3{1,i}(1,j);
                k=k+1;
                if k>8
                    break;
                end
            end
        end %for j=1:length(FP_3{1,i})      
    end %for i=1:24

%% problem4
    %% 生产能力
    ChanNeng=[]; Ta=0; Tb=0; Tc=0;
    [Fenpei_NUM,txt] = xlsread('附件1 近5年402家供应商的相关数据','供应商的供货量（m³）');
    txt4=txt(2:403,2);
    
    %% 求24周在10个周期中的最大值
    Gong_sum=sum(Gong);
    Gong_max_24=zeros(1,24);
    for i=1:24
        for j=1:10
            if Gong_sum(1,i+10*(j-1))>Gong_max_24(1,i)
                Gong_max_24(1,i)=Gong_sum(1,i+10*(j-1));
            end
        end
    end
    
    for i=1:24
       for j=1:402
           if txt4(j,1)=="A"
            Ta=Ta+Finishment(j,1)*Gong_max_24(1,i);
           end
           if txt4(j,1)=="B"
            Tb=Tb+Finishment(j,1)*Gong_max_24(1,i);
           end
           if txt4(j,1)=="C"
            Tc=Tc+Finishment(j,1)*Gong_max_24(1,i);
           end
       end
       Ta=Ta/Num_a;
       Tb=Tb/Num_b;
       Tc=Tc/Num_c;
       
       %% 订货（p4）
        Fenpei_4=num2cell(zeros(3,24));
        Fenpei_NUM_4=num2cell(zeros(3,24));
        Suma=0; Sumb=0; Sumc=0;
        a=[]; b=[]; c=[];
        an=[];bn=[]; cn=[];
        for i=1:24
            for j=1:Num_a
                if S_a(j,i)>6000
                    S_a(j,i)=6000;
                end
                Suma=Suma+S_a(j,i); 
                a(end+1)=S_a(j,i);
                an(end+1)=SNuma(j,1);
                if Suma>2*Ta
                    break;
                end
            end
            Fenpei_4{1,i}=a;
            Fenpei_NUM_4{1,i}=an;
            a=[];
            an=[];
            Suma=0;

            for j=1:Num_b
                if S_b(j,i)>6000
                    S_b(j,i)=6000;
                end
                Sumb=Sumb+S_b(j,i); 
                b(end+1)=S_b(j,i);
                bn(end+1)=SNumb(j,1);
                if Sumb>2*Tb
                    break;
                end
            end
            Fenpei_4{2,i}=b;
            Fenpei_NUM_4{2,i}=bn;
            b=[];
            bn=[];
            Sumb=0;

            for j=1:Num_c
                if S_c(j,i)>6000
                    S_c(j,i)=6000;
                end
                Sumc=Sumc+S_c(j,i);
                c(end+1)=S_c(j,i);
                cn(end+1)=SNumc(j,1); 
                if Sumc>2*Tc
                    break;
                end
            end
            Fenpei_4{3,i}=c;
            Fenpei_NUM_4{3,i}=cn;
            cn=[];
            c=[];
            Sumc=0;
        end
       
        %% 转运（p4）
        Fenpei_yun_4=num2cell(zeros(3,24));
        Fenpei_yun_num_4=num2cell(zeros(3,24));
        FPN_4=Fenpei_NUM_4;
        FP_4=Fenpei_4;

        for i=1:24
            FP_4(1,i)=strcat(FP_4(1,i),FP_4(2,i),FP_4(3,i));
            FPN_4(1,i)=strcat(Fenpei_NUM_4(1,i),Fenpei_NUM_4(2,i),Fenpei_NUM_4(3,i));
        end
        FP_4=FP_4(1,:);
        FPN_4=FPN_4(1,:);
        
        len=[];
        for i=1:24
            len(end+1)=length(FP_4{1,i});
        end
        for i=1:24
            Sum=0; a=[]; k=1; an=[];
            for j=1:len(i) 
                if FP_4{1,i}(1,j)>6000
                    FP_4{1,i}(1,j)=6000;
                end
                Sum=Sum+FP_4{1,i}(1,j); 
                a(end+1)=FP_4{1,i}(1,j);
                an(end+1)=FPN_4{1,i}(1,j);
                if Sum>6000
                    Fenpei_yun_4{k,i}=a(1:end-1);
                    Fenpei_yun_num_4{k,i}=an(1:end-1);
                    a=[];
                    an=[];
                    a(end+1)=FP_4{1,i}(1,j);
                    an(end+1)=FPN_4{1,i}(1,j);
                    Sum=FP_4{1,i}(1,j);
                    k=k+1;
                    if k>8
                        break;
                    end
                end
            end %for j=1:length(FP{1,i})      
        end %for i=1:24
        ChanNeng(end+1)=(Ta/0.6+Tb/0.66+Tc/0.72)*(1-betap(i)/100);
    end
    
    %% 提高的产能
    Gong_aver_total=sum(Gong_aver);
    delta_ChanNeng=[];
    for i=1:24
        delta_ChanNeng(end+1)=ChanNeng(i)-Gong_aver_total(i);
    end