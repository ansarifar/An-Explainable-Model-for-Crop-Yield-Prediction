clc
clear all
close all
%%
Pre_year=2018;
% seed='Corn';
seed='Soybeans';

[INFO_SOIL,INFO_Weather,INFO_PROGRESS,INFO_HARVESTED,INFO_Trend,INFO_ACRESPLANTED,...
                            INFO_Yield,Index_Train,Index_Valiation,Index_Test]=readdata(Pre_year,seed);
             

XX=[INFO_Weather,INFO_SOIL,INFO_Trend,INFO_PROGRESS,INFO_HARVESTED,INFO_ACRESPLANTED];

XX= (XX-min(XX))./(max(XX)-min(XX));
X_Train=XX(Index_Train,:);
X_Validation=XX(Index_Valiation,:);
X_Test=XX(Index_Test,:);
Y_Train=INFO_Yield(Index_Train,4);
Y_Validation=INFO_Yield(Index_Valiation,4);
Y_Test=INFO_Yield(Index_Test,4);


SOL.X_Train_UN=X_Train;
SOL.X_Validation_UN=X_Validation;
SOL.X_Test_UN=X_Test;
SOL.X_Train=X_Train;
SOL.X_Validation=X_Validation;
SOL.X_Test=X_Test;

SOL.Y_Train=Y_Train;
SOL.Y_Validation=Y_Validation;
SOL.Y_Test=Y_Test;



X_Train=XX(Index_Train,:);
X_Validation=XX(Index_Valiation,:);
X_Test=XX(Index_Test,:);
Y_Train=INFO_Yield(Index_Train,4);
Y_Validation=INFO_Yield(Index_Valiation,4);
Y_Test=INFO_Yield(Index_Test,4);

[RMSE,~,B]=Cal_RMSE(X_Train,Y_Train); 
disp(['The train RMSE: ',num2str(RMSE)])
disp(['The ratio of train error: ',num2str(RMSE/mean(Y_Train))])
Yp=[ones(numel(Y_Validation),1),X_Validation]*B;
RMSE=sqrt(mean((Y_Validation-Yp).*(Y_Validation-Yp)));
disp(['The Vali RMSE: ',num2str(RMSE)])
disp(['The ratio of Vali error: ',num2str(RMSE/mean(Y_Validation))])
Yp=[ones(numel(Y_Test),1),X_Test]*B;
RMSE=sqrt(mean((Y_Test-Yp).*(Y_Test-Yp)));
disp(['The test RMSE: ',num2str(RMSE)])
disp(['The ratio of test error: ',num2str(RMSE/mean(Y_Test))])
disp('=============================================')


SOL.X_Train_UN=X_Train;
SOL.X_Validation_UN=X_Validation;
SOL.X_Test_UN=X_Test;
SOL.X_Train=X_Train;
SOL.X_Validation=X_Validation;
SOL.X_Test=X_Test;

SOL.Y_Train=Y_Train;
SOL.Y_Validation=Y_Validation;
SOL.Y_Test=Y_Test;
             
COM=2;
Deep=2;
Threshold=0.0002;

SOL.HEU=Heuristic(SOL,COM,Deep,Threshold);

disp(['The train RMSE: ',num2str(SOL.HEU.RMSE_Train(end))])
disp(['The ratio of train error: ',num2str(SOL.HEU.RMSE_Validation(end)/mean(SOL.Y_Train))])

disp(['The Vali RMSE: ',num2str(SOL.HEU.RMSE_Validation(end))])
disp(['The ratio of Vali error: ',num2str(SOL.HEU.RMSE_Validation(end)/mean(SOL.Y_Validation))])

disp(['The test RMSE: ',num2str(SOL.HEU.RMSE_Test(end))])
disp(['The ratio of test error: ',num2str(SOL.HEU.RMSE_Test(end)/mean(SOL.Y_Test))])
disp('=============================================')
% 
find(SOL.HEU.Alfa~=0.5)
SOLL=SOL.HEU;
save('Interaction.mat','SOLL')
 %%
function [INFO_SOIL,INFO_Weather,INFO_PROGRESS,INFO_HARVESTED,INFO_Trend,INFO_ACRESPLANTED,INFO_Yield,Index_Train,Index_Valiation,Index_Test]=readdata(Pre_year,seed)

%load dataset.

if contains(seed, 'Corn')
    Start=1990; % Corn
    
    load INFO_ACRESPLANTED
    load INFO_HARVESTED
    load INFO_POPULATION
    load INFO_PROGRESS
    load INFO_SOIL
    load INFO_Weather
    load INFO_Yield    
else 
    Start=1990; %Soybeans 
    load INFO_ACRESPLANTED
    load INFO_HARVESTED
    load INFO_POPULATION
    load INFO_PROGRESS
    load INFO_SOIL
    load INFO_Weather
    load INFO_Yield      
end

Index_Train=find(INFO_Yield(:,1)>=Start  & INFO_Yield(:,1)<=(Pre_year-3) );
Index_Valiation=find(INFO_Yield(:,1)>=(Pre_year-2)  & INFO_Yield(:,1)<=(Pre_year-1) );
Index_Test=find(INFO_Yield(:,1)==Pre_year );

INFO_POPULATION=INFO_POPULATION(INFO_Yield(:,1)>=Start,:);
INFO_SOIL=INFO_SOIL(INFO_Yield(:,1)>=Start,4:end);
INFO_PROGRESS=INFO_PROGRESS(INFO_Yield(:,1)>=Start,4:end);
INFO_Weather=INFO_Weather(INFO_Yield(:,1)>=Start,136:end);
INFO_ACRESPLANTED=INFO_ACRESPLANTED(INFO_Yield(:,1)>=Start,4:end);
INFO_HARVESTED=INFO_HARVESTED(INFO_Yield(:,1)>=Start,4:end);
INFO_Yield=INFO_Yield(INFO_Yield(:,1)>=Start,:);

New_Fea1=zeros(size(INFO_Yield,1),1);
New_Fea2=zeros(size(INFO_POPULATION,1),1);
for i=[17,18,19]
    for j=1:205
        I=find(INFO_Yield(:,1)<Pre_year & INFO_Yield(:,2)==i & INFO_Yield(:,3)== j);
        Ip=find(INFO_Yield(:,2)==i & INFO_Yield(:,3)== j);
        IIp=INFO_Yield(Ip,:);
        II=INFO_Yield(I,:);
        II2=INFO_POPULATION(I,:);
        II(:,1)=II(:,1)-Start+1;
        II2(:,1)=II2(:,1)-Start+1;
        IIp(:,1)=IIp(:,1)-Start+1;
        beta=[ones(size(II,1),1),II(:,1)]\II(:,4);
        beta2=[ones(size(II2,1),1),II2(:,1)]\II2(:,4);
        New_Fea1(Ip)=[ones(size(IIp,1),1),IIp(:,1)]*beta;
        New_Fea2(Ip)=[ones(size(IIp,1),1),IIp(:,1)]*beta2;
    end
end


%load selected features.
if contains(seed, 'Corn')
    load FHar
    load FPro
    load FSoi
    load FWea
else
    load FHar
    load FPro
    load FSoi
    load FWea
end

INFO_Trend=[New_Fea1,New_Fea2];
INFO_SOIL=INFO_SOIL(:,FSoi);
INFO_PROGRESS=INFO_PROGRESS(:,FPro);
INFO_Weather=INFO_Weather(:,FWea);
INFO_HARVESTED=INFO_HARVESTED(:,FHar);

end

function [RMSE,Yp,Beta]=Cal_RMSE(X,Y)
         [N2,p]=size(X);
         Beta=([ones(N2,1),X+1e-6*eye(size(X))]\Y);
         Yp=[ones(N2,1),X]*Beta;
         RMSE=sqrt(mean((Yp-Y).*(Yp-Y)));       
end

function [sol,Index]=findindex(sol,num)

%            disp('----Feature selcetion starts----')
           F0=[]; 
           F1=[]; 
           F2=[]; 
           F3=[]; 
           F4=[];
           F5=[];
           beta=[ones(size(sol.X_Train_UN,1),1),sol.X_Train_UN]\sol.Y_Train;
           yhat=[ones(size(sol.X_Train_UN,1),1),sol.X_Train_UN]*beta;
           for i=1:size(sol.X_Train_UN,2)-1
               for j=i+1:size(sol.X_Train_UN,2)
                   
                   sigma_f=0.5;
                   sigma_n=0.5;
                   sigma_l=0.5;
                   alpha=0.5;
                   r=sol.X_Train_UN(:,i)-sol.X_Train_UN(:,j);     
                   Kernel0=r.*r;%sol.X_Train_UN(:,i).*sol.X_Train_UN(:,j); %Linear
                   Kernel1=(sigma_f^2)*exp(-0.5*(r.*r)./(sigma_l^2)); %SquaredExponential
                   Kernel2=(sigma_f^2)*exp(-(r.*r)/sigma_l); %Exponential 
                   Kernel3=(sigma_f^2)*(1+sqrt(3)*((r.*r)/sigma_l)).*exp(-sqrt(3)*(r)/sigma_l);  %Matern32  
                   Kernel4=(sigma_f^2)*(1+sqrt(5)*((r)/sigma_l)+(5/3)*(((r.*r)/sigma_l).^2)).*exp(-sqrt(5)*(r)/sigma_l); %Matern52
                   Kernel5=(sigma_f^2)*(1+(1/(2*alpha))*((r.*r)/sigma_l).^2).^(-alpha); %RationalQuadratic
                   h0=corrcoef(yhat-sol.Y_Train,Kernel0);
                   h1=corrcoef(yhat-sol.Y_Train,Kernel1);
                   h2=corrcoef(yhat-sol.Y_Train,Kernel2);
                   h3=corrcoef(yhat-sol.Y_Train,Kernel3);
                   h4=corrcoef(yhat-sol.Y_Train,Kernel4);
                   h5=corrcoef(yhat-sol.Y_Train,Kernel5);
                   F0=[F0;[abs(h0(1,2)),i,j]];
                   F1=[F1;[abs(h1(1,2)),i,j]];
                   F2=[F2;[abs(h2(1,2)),i,j]];
                   F3=[F3;[abs(h3(1,2)),i,j]];
                   F4=[F4;[abs(h4(1,2)),i,j]];
                   F5=[F5;[abs(h5(1,2)),i,j]];
               end
           end
           F=[F0;F1;F2;F3;F4;F5];
           Ker=[0*ones(length(F0),1);1*ones(length(F1),1);2*ones(length(F2),1);3*ones(length(F3),1);4*ones(length(F4),1);5*ones(length(F5),1)];
           [~,m]=sort(F(:,1),'descend');
           F=F(m,:);
           Ker=Ker(m,:);
           Ker(isnan(F(:,1)),:)=[];
           F(isnan(F(:,1)),:)=[];
           Ker(F(:,1)==1,:)=[];
           F(F(:,1)==1,:)=[];
           F=F(:,2:3)';
           F=unique(F(:),'stable');
            Index=F(1:num);
           sol.Xtr=sol.X_Train(:,Index);
           sol.Xte=sol.X_Test(:,Index);
           sol.Xva=sol.X_Validation(:,Index);
end

function [sol2,Index]=Heuristic(sol,Co,dp,Threshold)

       sol2.Alfa=[];
       Alfa_Total=[];
       sol2.Ef=0;
       NUMM=20;
       
       [sol,Index]=findindex(sol,NUMM);
       
       [sol2.B0, sol2.Bj, sol2.B, sol2.RMSE_Train, ~, sol2.Z_Train, sol2.KER] = ...
           RegEpi(sol.X_Train_UN, sol.Y_Train,sol.Xtr,sol2.Alfa);
       [sol2.RMSE_Validation, ~, sol2.Z_Validation]= RmseTest(sol.X_Validation_UN,...
           sol.Y_Validation,sol.Xva,sol2.B0, sol2.Bj, sol2.B,sol2.Alfa,sol2.KER);
       [sol2.RMSE_Test, ~, sol2.Z_Test]= RmseTest(sol.X_Test_UN,...
           sol.Y_Test,sol.Xte,sol2.B0, sol2.Bj, sol2.B,sol2.Alfa,sol2.KER);
       k=1;
       lams=0.5*ones(k,size(sol.Xtr,2));
       Ckeck=1;
       
       while Ckeck==1  && k<=10 
           [lams_All,rmse_All,rmse_All_validation,rmse_All_test,time_All,KER_All]=MutateEpi(Index,sol.X_Train,sol.Y_Train,...
               sol.X_Validation,sol.Y_Validation,sol.X_Test,sol.Y_Test,sol.Xtr,sol.Xva,sol.Xte,lams,Co*ones(1,1),dp);
           lams=lams_All(1,:,end);

           if sol2.RMSE_Validation(end)-rmse_All_validation(end)>=Threshold  

               J=find(lams(1,:)==1);
               J1=find(lams(1,:)==0);
               if numel(J)==0 && numel(J1)==0
                  ZK=[]; 
               else
                    if numel(J)==2
                        r=sol.Xtr(:,J(1))-sol.Xtr(:,J(2)); 
                        rv=sol.Xva(:,J(1))-sol.Xva(:,J(2));
                        rt=sol.Xte(:,J(1))-sol.Xte(:,J(2));  
                    elseif numel(J)==1 && numel(J1)==1
                        r=sol.Xtr(:,J(1))-(1-sol.Xtr(:,J1(1))); 
                        rv=sol.Xva(:,J(1))-(1-sol.Xva(:,J1(1))); 
                        rt=sol.Xte(:,J(1))-(1-sol.Xte(:,J1(1)));   
                    elseif numel(J1)==2
                        r=(1-sol.Xtr(:,J1(1)))-(1-sol.Xtr(:,J1(2))); 
                        rv=(1-sol.Xva(:,J1(1)))-(1-sol.Xva(:,J1(2))); 
                        rt=(1-sol.Xte(:,J1(1)))-(1-sol.Xte(:,J1(2))); 
                    elseif numel(J)==1 && numel(J1)==0
                        r=sol.Xtr(:,J(1))-sol.Xtr(:,J(1)); 
                        rv=sol.Xva(:,J(1))-sol.Xva(:,J(1)); 
                        rt=sol.Xte(:,J(1))-sol.Xte(:,J(1)); 
                    elseif numel(J)==0 && numel(J1)==1    
                        r=(1-sol.Xtr(:,J1(1)))-(1-sol.Xtr(:,J1(1))); 
                        rv=(1-sol.Xva(:,J1(1)))-(1-sol.Xva(:,J1(1))); 
                        rt=(1-sol.Xte(:,J1(1)))-(1-sol.Xte(:,J1(1))); 
                    end
                    sigma_f=0.5;
                    sigma_n=0.5;
                    sigma_l=0.5;
                    alpha=0.5; 
                    Kernel=zeros(size(r,1),6);
                    Kernel(:,1)=r.*r; %Linear
                    Kernel(:,2)=(sigma_f^2)*exp(-0.5*(r.*r)./(sigma_l^2)); %SquaredExponential
                    Kernel(:,3)=(sigma_f^2)*exp(-(r.*r)/sigma_l); %Exponential 
                    Kernel(:,4)=(sigma_f^2)*(1+sqrt(3)*((r.*r)/sigma_l)).*exp(-sqrt(3)*(r)/sigma_l);  %Matern32  
                    Kernel(:,5)=(sigma_f^2)*(1+sqrt(5)*((r)/sigma_l)+(5/3)*(((r.*r)/sigma_l).^2)).*exp(-sqrt(5)*(r)/sigma_l); %Matern52
                    Kernel(:,6)=(sigma_f^2)*(1+(1/(2*alpha))*((r.*r)/sigma_l).^2).^(-alpha); %RationalQuadratic
                    ZK(:,1)=Kernel(:,KER_All(end));
                    Kernel=zeros(size(rv,1),6);
                    Kernel(:,1)=rv.*rv; %Linear
                    Kernel(:,2)=(sigma_f^2)*exp(-0.5*(rv.*rv)./(sigma_l^2)); %SquaredExponential
                    Kernel(:,3)=(sigma_f^2)*exp(-(rv.*rv)/sigma_l); %Exponential 
                    Kernel(:,4)=(sigma_f^2)*(1+sqrt(3)*((rv.*rv)/sigma_l)).*exp(-sqrt(3)*(rv)/sigma_l);  %Matern32  
                    Kernel(:,5)=(sigma_f^2)*(1+sqrt(5)*((rv)/sigma_l)+(5/3)*(((rv.*rv)/sigma_l).^2)).*exp(-sqrt(5)*(rv)/sigma_l); %Matern52
                    Kernel(:,6)=(sigma_f^2)*(1+(1/(2*alpha))*((rv.*rv)/sigma_l).^2).^(-alpha); %RationalQuadratic
                    ZKv(:,1)=Kernel(:,KER_All(end));
                    Kernel=zeros(size(rt,1),6);
                    Kernel(:,1)=rt.*rt; %Linear
                    Kernel(:,2)=(sigma_f^2)*exp(-0.5*(rt.*rt)./(sigma_l^2)); %SquaredExponential
                    Kernel(:,3)=(sigma_f^2)*exp(-(rt.*rt)/sigma_l); %Exponential 
                    Kernel(:,4)=(sigma_f^2)*(1+sqrt(3)*((rt.*rt)/sigma_l)).*exp(-sqrt(3)*(rt)/sigma_l);  %Matern32  
                    Kernel(:,5)=(sigma_f^2)*(1+sqrt(5)*((rt)/sigma_l)+(5/3)*(((rt.*rt)/sigma_l).^2)).*exp(-sqrt(5)*(rt)/sigma_l); %Matern52
                    Kernel(:,6)=(sigma_f^2)*(1+(1/(2*alpha))*((rt.*rt)/sigma_l).^2).^(-alpha); %RationalQuadratic
                    ZKt(:,1)=Kernel(:,KER_All(end));
               end
               

               sol.X_Train=[sol.X_Train,ZK];
               sol.X_Validation=[sol.X_Validation,ZKv];
               sol.X_Test=[sol.X_Test,ZKt];
               
               Alfa_Total=[0.5*ones(1,size(sol.X_Train_UN,2))];
               Alfa_Total(1,Index((lams_All(1,:,end)==1)))=1;
               Alfa_Total(1,Index((lams_All(1,:,end)==0)))=0;

               
               if NUMM<=30
                  NUMM=NUMM+10;
               end
               [sol,Index]=findindex(sol,NUMM);
               
                lams=0.5*ones(1,size(sol.Xtr,2));
                
                sol2.Alfa(k,:)=Alfa_Total;
%                 sol2.T=[sol2.T,sol2.T(end)+time_All];
                sol2.Ef=[sol2.Ef,repmat(k,1,numel(time_All))];
                sol2.RMSE_Train=[sol2.RMSE_Train,rmse_All];
                sol2.RMSE_Validation=[sol2.RMSE_Validation,rmse_All_validation];
                sol2.RMSE_Test=[sol2.RMSE_Test,rmse_All_test];
                sol2.KER(k)=KER_All(end);
                k=k+1;           
             else
                Ckeck=0; 
%                 sol2.T=[sol2.T,sol2.T(end)+time_All(end)];
                sol2.Ef=[sol2.Ef,sol2.Ef(end)];
                sol2.RMSE_Train=[sol2.RMSE_Train,sol2.RMSE_Train(end)];
                sol2.RMSE_Validation=[sol2.RMSE_Validation,sol2.RMSE_Validation(end)];
                sol2.RMSE_Test=[sol2.RMSE_Test,sol2.RMSE_Test(end)];                
           end         
       end
       
       [sol2.B0, sol2.Bj, sol2.B, R1, ~, sol2.Z_Train,sol2.KER] = ...
               RegEpi2(sol.X_Train_UN, sol.Y_Train,sol2.Alfa);
       [R2, ~, sol2.Z_Validation]= RmseTest2(sol.X_Validation_UN,...
               sol.Y_Validation,sol2.B0, sol2.Bj, sol2.B,sol2.Alfa,sol2.KER);  
       [R3, ~, sol2.Z_Test]= RmseTest2(sol.X_Test_UN,...
               sol.Y_Test,sol2.B0, sol2.Bj, sol2.B,sol2.Alfa,sol2.KER);     
end
function [lams_All,rmse_All,rmse_All_validation,rmse_All_test,time_All,KER_All] = MutateEpi(Index,X_Train,Y_Train,X_Validation,Y_Validation,X_Test,Y_Test,Xtr,Xva,Xte, lams0, CS, dp)%, timelimit)
  dispm = 1;
  fig = 0;
  impatient = 0;

  
  [K, p] = size(lams0);
  lams_best = lams0;
  [beta0, betap0, betaepi, rmse_best,~,~,KER_best] = RegEpi(X_Train,Y_Train,Xtr,lams0);
  [rmse_validation, ~, ~]= RmseTest(X_Validation,Y_Validation,Xva,beta0,betap0,betaepi,lams0,KER_best);
  [rmse_test, ~, ~]= RmseTest(X_Test,Y_Test,Xte,beta0,betap0,betaepi,lams0,KER_best);

  betap_best = betap0;
  nall = 1;
  lams_All = lams_best;
  KER_All = KER_best;
  rmse_All = rmse_best;
  rmse_All_validation=rmse_validation;
  rmse_All_test=rmse_test;
  time_All = 0;

  if nargin <= 7
    timelimit = 300;
  end
  if nargin <= 6
    dp = 2;
  end
  if nargin <= 5
    for k = 1:K
      CS(k) = sum(lams0(k,:)~=0.5);
    end
  end
  for i = 1:dp
    permd{i} = nchoosek(1:p,i);
    perm01{i} = dec2base(0:2^i-1,2)-47;
  end
  if fig == 1
    figure;
    fxs = 0;
    fys = rmse_best;
  end

  okd = zeros(K,dp);
  d = 1;
  k = 1;
  k0 = 1;
  tic;
  
  timelimit=1000000000000;
  while toc < timelimit - 1e-4
    fkd = find(okd<0.5,1);
    if isempty(fkd)
      lams_All(:,:,nall+1) = lams_best;
      KER_All(:,:,nall+1) = KER_best;
      rmse_All(nall+1) = rmse_best;
      rmse_All_validation(nall+1) = rmse_validation;
      rmse_All_test(nall+1) = rmse_test;
      time_All(nall+1) = toc; 
      return;
    end
    [k, d] = ind2sub([K dp],fkd);
    if okd(k0, d) < 0.5
      k = k0;
    end

    if dispm == 1
      fprintf('Working on k = %d and d = %d \n',k,d);
    end

    if fig == 1
      fxs = [fxs; toc];
      fys = [fys; rmse_best];
      plot(fxs,fys);
      xlim([0 timelimit]);
      ylim([0 1.01*rmse_All(1)]);
      xlabel('Time');
      ylabel('RMSE');
      drawnow;
    end

    [lams_d, rmse_d, beta_d] = MutateEpi_1k1dp(X_Train,Y_Train,Xtr, lams_best, CS, k, rmse_best, permd{d}, perm01{d}, timelimit, impatient);
    [b0p, bp0p, bpip, ~,~,~,KER_d] = RegEpi(X_Train,Y_Train,Xtr, lams_d);
    [rmse_d1, ~, ~]= RmseTest(X_Validation,Y_Validation,Xva,b0p, bp0p, bpip,lams_d,KER_d);
    if rmse_d1 <rmse_validation
      if dispm == 1
        fprintf('RMSE %f, Time %f.\n',rmse_d1,toc);
      end
      lams_best = lams_d;
      rmse_best = rmse_d;
      betap_best = beta_d;
      KER_best=KER_d;
      [~, isbeta] = sort(abs(betap0(Index)-betap_best(Index)),'descend');
      for i = 1:dp
        permd{i} = nchoosek((isbeta),i);
      end

      lams_All(:,:,nall+1) = lams_best;
      rmse_All(nall+1) = rmse_best;
      time_All(nall+1) = toc;
      KER_All(:,:,nall+1) = KER_best;
      [beta0p, betap0p, betaepip, ~,~,~,KER] = RegEpi(X_Train,Y_Train,Xtr, lams_best);
      [rmse_test, ~, ~]= RmseTest(X_Test,Y_Test,Xte,beta0p, betap0p, betaepip,lams_best,KER);  
      [rmse_validation, ~, ~]= RmseTest(X_Validation,Y_Validation,Xva,beta0p, betap0p, betaepip,lams_best,KER);
            
      rmse_All_validation(nall+1) = rmse_validation;
      rmse_All_test(nall+1) = rmse_test;
      
      nall = nall + 1;
      okd(:,1:d) = 0;
      k0 = k;
    else
      okd(k,d) = 1;
    end
  end

  if fig == 1
    fxs = [fxs; toc];
    fys = [fys; rmse_best];
    plot(fxs,fys);
    xlim([0 timelimit]);
    ylim([0 1.01*rmse_All(1)]);
    xlabel('Time');
    ylabel('RMSE');
  end
end

function [lams, rmse, betap] = MutateEpi_1k1dp(X, y,Xtr, lams, CS, k, rmse, permd, perm01, timelimit, impatient)
  [K, p] = size(lams);
  [npd, dp] = size(permd);
  np01 = size(perm01,1);
  p012 = zeros(2,p);
  p012(1, lams(k,:)==0) = 0.5;
  p012(2, lams(k,:)==0) = 1;
  p012(1, lams(k,:)==0.5) = 0;
  p012(2, lams(k,:)==0.5) = 1;
  p012(1, lams(k,:)==1) = 0;
  p012(2, lams(k,:)==1) = 0.5;
  vs = inf(npd,np01);
  lamk0 = lams(k,:);
  betap = zeros(p,1);
  for i = 1:npd
    p012perm = p012(:,permd(i,:));
    for j = 1:np01
      if toc >= timelimit
        break;
      end
      lams(k,:) = lamk0;
      lams(k,permd(i,:)) = p012perm(sub2ind([2 dp],perm01(j,:),1:dp));
      if sum(lams(k,:)~=0.5) <= CS(k)
        [~, betaps(:,i,j), ~, vs(i,j)] = RegEpi(X, y,Xtr, lams);
      end
      if and(impatient == 1, vs(i,j) < rmse)
        rmse = vs(i,j);
        betap = betaps(:,i,j);
        return;
      end
    end
  end
  [minv, iminv] = min(vs(:));
  lams(k,:) = lamk0;
  if minv < rmse - 1e-6
    [i,j] = ind2sub([npd,np01], iminv);
    p012perm = p012(:,permd(i,:));
    lams(k,permd(i,:)) = p012perm(sub2ind([2 dp],perm01(j,:),1:dp));
    rmse = minv;
    betap = betaps(:,i,j);
  end
end

function [rmse, yh, ZK]= RmseTest(X,y,Xte,beta0,betap,betaepi,lams,KER)

  warning ('off');
  [n, p] = size(Xte);
  if nargin <= 4
    lams = zeros(0,p);
    K = 0;
    ZK = zeros(n,0);
    betaepi = 0;
  else
    K = size(lams,1);
    ZK = zeros(n,K);
    for k = 1:K
        J=find(lams(k,:)==1);
        J1=find(lams(k,:)==0);
        if numel(J)==0 && numel(J1)==0
        else
            if numel(J)==2
                r=Xte(:,J(1))-Xte(:,J(2));  
            elseif numel(J)==1 && numel(J1)==1
                r=Xte(:,J(1))-(1-Xte(:,J1(1)));    
            elseif numel(J1)==2
                r=(1-Xte(:,J1(1)))-(1-Xte(:,J1(2)));    
            elseif numel(J)==1 && numel(J1)==0
                r=Xte(:,J(1));  
            elseif numel(J)==0 && numel(J1)==1    
                r=(1-Xte(:,J1(1)));  
            end
            sigma_f=0.5;
            sigma_n=0.5;
            sigma_l=0.5;
            alpha=0.5;
            Kernel=zeros(n,6);
            Kernel(:,1)=r.*r;%sol.X_Train_UN(:,i).*sol.X_Train_UN(:,j); %Linear
            Kernel(:,2)=(sigma_f^2)*exp(-0.5*(r.*r)./(sigma_l^2)); %SquaredExponential
            Kernel(:,3)=(sigma_f^2)*exp(-(r.*r)/sigma_l); %Exponential 
            Kernel(:,4)=(sigma_f^2)*(1+sqrt(3)*((r.*r)/sigma_l)).*exp(-sqrt(3)*(r)/sigma_l);  %Matern32  
            Kernel(:,5)=(sigma_f^2)*(1+sqrt(5)*((r)/sigma_l)+(5/3)*(((r.*r)/sigma_l).^2)).*exp(-sqrt(5)*(r)/sigma_l); %Matern52
            Kernel(:,6)=(sigma_f^2)*(1+(1/(2*alpha))*((r.*r)/sigma_l).^2).^(-alpha); %RationalQuadratic
            ZK(:,k)=Kernel(:,KER(k));
        end
    end
  end
  yh = beta0 + X * betap+ ZK * betaepi ;
  rmse = sqrt((y-yh)'*(y-yh)/n);             
end

function [beta0, betap, betaepi, rmse, yh, ZK,KER] = RegEpi(X,y,Xtr,lams)


 rmsep = inf; 



  warning ('off');
  [n, p] = size(Xtr);
  if nargin <= 2
    lams = zeros(0,p);
    K = 0;
    ZK = zeros(n,0);
    KER = zeros(0,K);
    betaepi = [];
  else
    K = size(lams,1);
    betaepi = zeros(K,1);
    ZK = zeros(n,K);
    KER = zeros(1,K);
    for k = 1:K
        J=find(lams(k,:)==1);
        J1=find(lams(k,:)==0);
        if numel(J)==0 && numel(J1)==0
        else
            if numel(J)==2
                r=Xtr(:,J(1))-Xtr(:,J(2));  
            elseif numel(J)==1 && numel(J1)==1
                r=Xtr(:,J(1))-(1-Xtr(:,J1(1)));    
            elseif numel(J1)==2
                r=(1-Xtr(:,J1(1)))-(1-Xtr(:,J1(2)));    
            elseif numel(J)==1 && numel(J1)==0
                r=Xtr(:,J(1));  
            elseif numel(J)==0 && numel(J1)==1    
                r=(1-Xtr(:,J1(1)));  
            end
            sigma_f=0.5;
            sigma_n=0.5;
            sigma_l=0.5;
            alpha=0.5;
            Kernel=zeros(n,6);
            Kernel(:,1)=r.*r;%sol.X_Train_UN(:,i).*sol.X_Train_UN(:,j); %Linear
            Kernel(:,2)=(sigma_f^2)*exp(-0.5*(r.*r)./(sigma_l^2)); %SquaredExponential
            Kernel(:,3)=(sigma_f^2)*exp(-(r.*r)/sigma_l); %Exponential 
            Kernel(:,4)=(sigma_f^2)*(1+sqrt(3)*((r.*r)/sigma_l)).*exp(-sqrt(3)*(r)/sigma_l);  %Matern32  
            Kernel(:,5)=(sigma_f^2)*(1+sqrt(5)*((r)/sigma_l)+(5/3)*(((r.*r)/sigma_l).^2)).*exp(-sqrt(5)*(r)/sigma_l); %Matern52
            Kernel(:,6)=(sigma_f^2)*(1+(1/(2*alpha))*((r.*r)/sigma_l).^2).^(-alpha); %RationalQuadratic
            for l=1:6
               ZK(:,k)=Kernel(:,l); 
               br = [ones(n,1),X,ZK(:,1:k)]\y;
               yh = [ones(n,1),X,ZK(:,1:k)]*br;
               rmse = sqrt((y-yh)'*(y-yh)/n) ;
               if rmsep>= rmse
                  best=l; 
                  rmsep=rmse;
               end 
            end
            ZK(:,k)=Kernel(:,best); 
            KER(k)=best;
        end
    end
  end
  [n, p] = size(X);
  br = [ones(n,1),X,ZK]\y;
  beta0 = br(1);
  betap = br(2:p+1);
  betaepi = br(p+1+[1:K]);
  yh = [ones(n,1),X,ZK]*br;
  rmse = sqrt((y-yh)'*(y-yh)/n);
end

function [beta0, betap, betaepi, rmse, yh, ZK,KER] = RegEpi2(X, y, lams)
  

 rmsep = inf; 

 warning ('off');
  [n, p] = size(X);
  if nargin <= 2
    lams = zeros(0,p);
    K = 0;
    ZK = zeros(n,0);
    KER = zeros(0,K);
    betaepi = [];
  else
    K = size(lams,1);
    betaepi = zeros(K,1);
    ZK = zeros(n,K);
    KER = zeros(1,K);
    for k = 1:K
        J=find(lams(k,:)==1);
        J1=find(lams(k,:)==0);
        if numel(J)==0 && numel(J1)==0
        else
            if numel(J)==2
                r=X(:,J(1))-X(:,J(2));  
            elseif numel(J)==1 && numel(J1)==1
                r=X(:,J(1))-(1-X(:,J1(1)));    
            elseif numel(J1)==2
                r=(1-X(:,J1(1)))-(1-X(:,J1(2)));    
            elseif numel(J)==1 && numel(J1)==0
                r=X(:,J(1));  
            elseif numel(J)==0 && numel(J1)==1    
                r=(1-X(:,J1(1)));  
            end
            sigma_f=0.5;
            sigma_n=0.5;
            sigma_l=0.5;
            alpha=0.5;
            Kernel=zeros(n,6);
            Kernel(:,1)=r.*r;%sol.X_Train_UN(:,i).*sol.X_Train_UN(:,j); %Linear
            Kernel(:,2)=(sigma_f^2)*exp(-0.5*(r.*r)./(sigma_l^2)); %SquaredExponential
            Kernel(:,3)=(sigma_f^2)*exp(-(r.*r)/sigma_l); %Exponential 
            Kernel(:,4)=(sigma_f^2)*(1+sqrt(3)*((r.*r)/sigma_l)).*exp(-sqrt(3)*(r)/sigma_l);  %Matern32  
            Kernel(:,5)=(sigma_f^2)*(1+sqrt(5)*((r)/sigma_l)+(5/3)*(((r.*r)/sigma_l).^2)).*exp(-sqrt(5)*(r)/sigma_l); %Matern52
            Kernel(:,6)=(sigma_f^2)*(1+(1/(2*alpha))*((r.*r)/sigma_l).^2).^(-alpha); %RationalQuadratic
            for l=1:6
               ZK(:,k)=Kernel(:,l); 
               br = [ones(n,1),X,ZK(:,1:k)]\y;
               yh = [ones(n,1),X,ZK(:,1:k)]*br;
               rmse = sqrt((y-yh)'*(y-yh)/n) ;
               if rmsep>= rmse
                  best=l; 
                  rmsep=rmse;
               end 
            end
            ZK(:,k)=Kernel(:,best); 
            KER(k)=best;
        end
    end
  end
  [n, p] = size(X);
  br = [ones(n,1),X,ZK]\y;
  beta0 = br(1);
  betap = br(2:p+1);
  betaepi = br(p+1+[1:K]);
  yh = [ones(n,1),X,ZK]*br;
  rmse = sqrt((y-yh)'*(y-yh)/n);
end

function [rmse, yh, ZK]= RmseTest2(X,y,beta0,betap,betaepi,lams,KER)

  warning ('off');
  [n, p] = size(X);
  if nargin <= 4
    lams = zeros(0,p);
    K = 0;
    ZK = zeros(n,0);
    betaepi = 0;
  else
    K = size(lams,1);
    ZK = zeros(n,K);
    for k = 1:K
        J=find(lams(k,:)==1);
        J1=find(lams(k,:)==0);
        if numel(J)==0 && numel(J1)==0
        else
            if numel(J)==2
                r=X(:,J(1))-X(:,J(2));  
            elseif numel(J)==1 && numel(J1)==1
                r=X(:,J(1))-(1-X(:,J1(1)));    
            elseif numel(J1)==2
                r=(1-X(:,J1(1)))-(1-X(:,J1(2)));    
            elseif numel(J)==1 && numel(J1)==0
                r=X(:,J(1));  
            elseif numel(J)==0 && numel(J1)==1    
                r=(1-X(:,J1(1)));  
            end
            sigma_f=0.5;
            sigma_n=0.5;
            sigma_l=0.5;
            alpha=0.5;
            Kernel=zeros(n,6);
            Kernel(:,1)=r.*r;%sol.X_Train_UN(:,i).*sol.X_Train_UN(:,j); %Linear
            Kernel(:,2)=(sigma_f^2)*exp(-0.5*(r.*r)./(sigma_l^2)); %SquaredExponential
            Kernel(:,3)=(sigma_f^2)*exp(-(r.*r)/sigma_l); %Exponential 
            Kernel(:,4)=(sigma_f^2)*(1+sqrt(3)*((r.*r)/sigma_l)).*exp(-sqrt(3)*(r)/sigma_l);  %Matern32  
            Kernel(:,5)=(sigma_f^2)*(1+sqrt(5)*((r)/sigma_l)+(5/3)*(((r.*r)/sigma_l).^2)).*exp(-sqrt(5)*(r)/sigma_l); %Matern52
            Kernel(:,6)=(sigma_f^2)*(1+(1/(2*alpha))*((r.*r)/sigma_l).^2).^(-alpha); %RationalQuadratic
            ZK(:,k)=Kernel(:,KER(k));
        end
    end
  end
  yh = beta0 + X * betap+ ZK * betaepi ;
  rmse = sqrt((y-yh)'*(y-yh)/n);              
end
