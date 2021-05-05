% 
% This file includes code for the paper 
% Online Aggregation of Probability Forecasts with Confidence. 
% @authors: Vladimir Vyugin, Vladimir Trunov
% 

function [ output_args ] = run_synth_data_experiments()
% Execute artificial model experiments ni_crps
% 1 - "shifted" generating models
[r] = triangle_dist_model('a',1,0.001);
% 2 - mixed generating models
[r] = triangle_dist_model('b',2,0.001);
return;

function [ output_args ] = triangle_dist_model(tag,model_variant,fs_alpha)
% To create FIG and EPS figures, set paper_figures to true!
paper_figures = false;
chars ={'A' 'B' 'C' 'D'};
figure_path=['..\figures1\'];
if nargin==0
    tag = 'DBG'
    model_variant = 2;
    fs_alpha      = 0.001;
end;
output_args = 0;
LineSpecList1 = {'--b','--g','--r',':b',':g',':r'};
LineSpecList2 = {'-b','-g','-r',':b',':g',':r'};

verbose = 0;          
LL = 1000;            
alpha = fs_alpha;     % Fixed Share
lowerX = -10;         % [a b]  
upperX =  10;
%% Support interval
z =  lowerX:0.01:upperX; 
eta_v = 2/(upperX-lowerX);    % Online correction of expert weighta for spread [a,b]
eta_w = 0.5/(upperX-lowerX);  
eta2  = 2;          % For Vovk "superprediction"  

model_prefix = ...
  ['TModel-' num2str(model_variant) '/' num2str(LL) '/Fs=' num2str(alpha)];

%% Default settings
L = 100;                % Segment lengths
e_peak = [-5 0 5];      % Central points
e_spread = [9.5 6 9.5]; 
N = length(e_spread);   
B = [1 2 3 1 2 3];      % Leaders
S = 6; % 
T = S*L;                
jittered_priors = 0;
 
Priors = zeros(length(e_peak),S*L);
switch model_variant
    case 1  % One shifting expert
        for s=1:S
            leader = B(s);
            Priors(leader,(s-1)*L+1:(s*L)) = 1;
        end
    case 2
        % Mix of 3 expert distributions   
        PPP=w_explore(T,2,10);
        Priors = PPP';
        %Reversing of Priors
        Pr = Priors;
        for t=1:T
            Priors(:,t) = Pr(:,T-t+1);
        end
    otherwise
        error('Illegal model_variant!');
end
PPP = Priors';
       
%% Create generating models and signal
%% PPLN - PoPuLatioN, constructed by generating models
PPLN = zeros(N,T);      
experts = cell(1,length(e_peak));    % experts
rng(0);

for i=1:length(e_peak)
    if i==1
       e_struct.peak     = -8;
       e_struct.spread   = 10.5;
       e_struct.a = -9.5;
       e_struct.b = -8;
       e_struct.c =  1;
    elseif i==2
       e_struct.peak     = 0;
       e_struct.spread   = 10;
       e_struct.a = -5;
       e_struct.b = -0;
       e_struct.c =  5; 
    elseif i==3        
       e_struct.peak     = 8;
       e_struct.spread   = 10.5;
       e_struct.a = -1;
       e_struct.b =  8;
       e_struct.c =  9.5;
    end   
    e_struct.pd = ...
       makedist('Triangular','a',e_struct.a,'b',e_struct.b,'c',e_struct.c); 
    sample_length = LL;
    e_struct.sample = random(e_struct.pd,sample_length,1); % Column 
    PPLN(i,:) = random(e_struct.pd,1,T);                   % Line
    experts{i}   = e_struct;
end

% if verbose
%     % Draw PPLN
%     figure;
%     %colormap(jet)
%     draw_bgr_lines(PPLN);
%     %plot(PPLN');
%     q=1;
% end

w0 = ones(N,1);
w0 = w0/sum(w0);
w  = w0;
U  = [];               % Time series
BB = [];               % Leader

%% Create time series U(1:T) using Priors and PoPuLatioN 
for t=1:T
    pt = Priors(:,t);       % Piors 
    omega = get_coin(pt);   % omega
    U(t)=PPLN(omega,t);     % Select generator
end
disp([lowerX,upperX]);      % Selected spread
disp([min(U),max(U)])       % Actual spread 
q=1;

% Calculate CRPS values of N experts in all points of U
CRPS   = nan(N,T);
smCRPS = nan(N,T);
for i=1:N
    ff = cdf(experts{i}.pd,z);
    %Dens(1,:) = pdf(experts{i}.pd,z);
    for t=1:T           % CRPS calculation
        u=U(t);
        CRPS(i,t) = ni_crps(ff,z,u);      
    end
end

% Online aggregation  
% Wrm,Vvk: CRPS using empirical expert distributions 
% Eta - eta_v/eta_w  (Vovk/Warmuth) %% FS=FixedShare
[WWev0,Laggr_ev0] = apply_crps_mixing(U,CRPS,eta_v,0,experts); %V No FS!
[WWew0,Laggr_ew0] = apply_crps_mixing(U,CRPS,eta_w,0,experts); %W No FS!
[WWev,Laggr_ev]   = apply_crps_mixing(U,CRPS,eta_v,alpha,experts); %V+FS
[WWew,Laggr_ew]   = apply_crps_mixing(U,CRPS,eta_w,alpha,experts); %W+FS

% WWth - adapted weights of experts
% Laggr_th - weighted sum of expert CRPS losses

hh = draw_sublot_figure_plus(PPLN,PPP,U,WWev',WWew',model_prefix);
if paper_figures
    saveas(hh,[figure_path 'fig2' tag '.fig']);
    saveas(gcf,[figure_path 'fig2' tag],'epsc');
end
saveas(hh,[figure_path 'fig2' tag '.png']);

% Online aggregation using superprediction (Vovk).
% WW_ev calculate on the base of empical expert CDF
% Do not depend on t !
A   = zeros(N,length(z));  % -eta_v*F(i,t)^2
B   = zeros(N,length(z));  % -eta_v*(1-*F(i,t))^2
FF  = zeros(1,length(z));  % Master Expert CDF
FFw = zeros(1,length(z));  % Сonvex sum of expert CDF 
C   = zeros(N,length(z));  % Expert CDF at points of z
% Create matrix (A) and (B)
for n=1:N
    pd = experts{n}.pd;    % empical  CDF of expert n
    for i = 1:length(z)    % for all points 
        u = z(i);
        Fn = cdf(pd,u);

        A(n,i) = exp(-eta2*Fn^2);         % Nominator
        B(n,i) = exp(-eta2*(1-Fn)^2);     % Denominator
        C(n,i) = cdf(pd,u);
    end
end

CRPSv = zeros(1,T); % CRPS for superprediction 
CRPSw = zeros(1,T); % For WRM... 

%% Calc.CRPSv and (CRPSw) 
draw_3d_picture = 1; %true;
if draw_3d_picture
    figure; hold on
end
for t = 1:T
    u = U(t);       
    % Form Ft using A,B, and Wt 
    Wt = WWev(:,t);      
    Vn = A'*Wt;             %  numerator
    Vd = B'*Wt;             %  denomerator
    R  = Vn ./ Vd;          %  Devide
    %FF = 0.5*(1-(1/eta_v)*log(R)); 
    FF = 0.5*(1-(1/2)*log(R)); 
    FF(1) = 0;             
    FF(end) = 1;
    %FFpd = makedist('piecewiselinear',z,FF'); % Master eCDF
    
    FFw  = (WWew(:,t)')*C;  % Mix by Wrm
    
    if draw_3d_picture
        if (mod(t,2)==0)
            plot3(t*ones(1,length(z)),z,FF');
        end
    end
    CRPSw(t) = ni_crps(FFw,z,u); 
    CRPSv(t) = ni_crps(FF',z,u); % Master CRPS for u
end
if draw_3d_picture
    set(gcf);
    xlim([0 T]);
    view(45,45);
    grid on;
    box on;       % saveas(gcf,[fig3 tag '.fig'])
    if paper_figures
        saveas(gcf,[figure_path 'fig4' tag '.fig']);
        saveas(gcf,[figure_path 'fig4' tag],'epsc');
    end
    saveas(gcf,[figure_path 'fig4' tag '.png']);
    %saveas(gcf,[tag '-' model_label '3dCDF_v.fig']);
end
%save(['rrr_' tag '.mat'],'CRPS',...
%     'CRPSv','CRPSw','Laggr_ev','Laggr_ew','WWev','WWew','Priors');
q=1;

if true
    figure;
    LW = 2;
    plot(cumsum(CRPS'),'LineWidth',LW); hold on;  % Experts...
    %LLCC=[cumsum(Laggr_ew) cumsum(Laggr_ev) cumsum(CRPSv') cumsum(CRPSw')];
    LLCC=[cumsum(CRPSv') cumsum(CRPSw')];
    plot(LLCC(:,1),':r','LineWidth',LW);
    plot(LLCC(:,2),'--b','LineWidth',LW);
    %legend('1','2','3','Laggr-eW','Laggr-eV','Vovk','Warmuth');
    %title([model_prefix 'Cumulated CRPS Losses. LL=' num2str(LL) '.']);
    q=1;
    grid on
    if paper_figures
        saveas(gcf,[figure_path 'fig3' tag '.fig']);
        saveas(gcf,[figure_path 'fig3' tag],'epsc');
    end
    saveas(gcf,[figure_path 'fig3' tag '.png']);
end

function [WW,Laggr] = apply_crps_mixing(U,CRPS,eta,alpha,exp_params)
% U - time series
% CRPS  - matrix of CRPS at U points, eta-train param.
% alpha - fixed_share
% exp_params - params of experts
% Laggr - linear combination of experts CRPS 
[N,T]=size(CRPS);
w0 = ones(N,1);
w0 = w0/sum(w0);
WW(:,1)=w0;
w  = w0;

for t=1:T-1
    for n=1:N
        w(n)=w0(n)*exp(-eta*CRPS(n,t));
    end
    w = w/sum(w);
    % Fixed share
    w = alpha/N+(1-alpha)*w;
    
    w0 = w;  
    WW(:,t+1)=w;
end

Laggr = nan(T,1);
for i=1:T
    Laggr(i) = WW(:,i)' * CRPS(:,i);
end

function s = ni_crps(f,z,u)
% f - linear approximation of distribution
% z - grid points of argument
% u - test point
% Calculation of CRPS for empirical distribution with compact support
% using integration by trapezoidal method. The integral can be turned 
% into discrete finite sum.
dz = diff(z);  
n=length(z);
s=0;
if u<=z(1)      % (1-F)^2 
    for t=1:n-1
        s=s+dz(t)*((f(t)+f(t+1))^2)*0.25;
    end
elseif u>=z(n)  
    for t=1:n-1
        s=s+dz(t)*(1-(f(t)+f(t+1))/2)^2;
    end
else
    ind = find(z<=u,1,'last'); 
    for t=1:ind-1
        s=s+dz(t)*0.25*(f(t)+f(t+1))^2;
    end
    t=ind;
    d1 = u-z(t);
    d2 = z(t+1)-u;
    d  = z(t+1)-z(t);  
    fu = f(t)+(f(t+1)-f(t))*(d1/(d1+d2));
    fu1= (f(t)+fu)*0.5;
    fu2= (fu+f(t+1))*0.5;
    s = s + d1*(fu1)^2+d2*(1-fu2)^2;
    for t=ind+1:n-1
        s=s+dz(t)*(1-f(t))^2;
    end
end


function r = get_coin(pt)
% randomization 
d=rand;
w=cumsum(pt);
r=find(d<=w,1,'first');
    
function hh = draw_sublot_figure_plus(PPLN,PPP,U,PM1,PM2,txt)
    dd=2;
    [N,T]=size(PPLN);
    %colormap(jet);
    hh=figure('Name',txt);
    subplot(5,1,1);
    plot(PPLN');
    %draw_bgr_lines(PPLN,0.5);
    xlim([0 T]);
    %ylabel('A');  %XXX
    ylabel(gca,['A '],'FontSize',12,'FontWeight','bold','rotation',0);
    %title('Generative Models')
    %colormap(jet);
    subplot(5,1,2);
    area(PPP);
    xlim([0 T]);
    ylim([0 1]);
    %ylabel('B');  %XXX
    ylabel(gca,['B '],'FontSize',12,'FontWeight','bold','rotation',0);
    %title('Apriory Mixture Proportions');
    subplot(5,1,3);
    plot(U); hold on;
    %uu = smooth(U,71);
    %plot(uu,':r','LineWidth',1);
    grid on;
    %title(['Mixture of generative models.']);
    %ylabel('C'); %XXX
    ylabel(gca,['C '],'FontSize',12,'FontWeight','bold','rotation',0);
    xlim([0 T]);
    subplot(5,1,4);
%     title('Posterior Mixture Proportions (area plot)');
    nn = size(PM1,1); 
    x=1:nn';
    % Not all points
    ind=(1:dd:nn)';
    xx=x(ind);
    PMs=PM1(ind,:);
    area(xx,PMs);
    %area(PM);
    xlim([0 T]);
    ylim([0 1]);
    %ylabel('D');
    ylabel(gca,['D '],'FontSize',12,'FontWeight','bold','rotation',0);
    %xlabel('Time');
    %title(txt);
    subplot(5,1,5);
%    title('Posterior Mixture Proportions (area plot)');
    x=1:nn';
    ind=(1:dd:nn)';
    xx=x(ind);
    PMs=PM2(ind,:);
    area(xx,PMs);
    %area(PM);
    xlim([0 T]);
    ylim([0 1]);
    %ylabel('E');
    ylabel(gca,['E '],'FontSize',12,'FontWeight','bold','rotation',0);
    %xlabel('Time');
    %title(txt);
    %colormap(jet);
    q=1;    

function Y = jitt(X,sc)
    if nargin==1
        sc=1;
    end
    noise = randn(size(X))*sc;
    Y=X+noise;
    
function [PPP] = w_explore(T,ind,scale)
% Synthesize component weights (priors)
if nargin==0
    T=1200;
    ind = 4;
    scale = 1;
elseif nargin<3
    scale = 1;
end
omega=scale*2*pi/4000;
s1=100;
s2=250;
s3=1000; %125
t=(1:T)';
tt= t .* t;
w=zeros(length(t),3);
switch ind
    case 0
        w(:,1) = 0.333*(1+t/6000);
        w(:,2) = 0.333+sin(t*omega/2)*0.333;
        w(:,3) = 0.333*(1-t/6000);
    case 1
        w(:,1) = sin(t*omega+s1)+1.21;
        w(:,2) = sin(t*omega*1.5+s2)+1.01;
        w(:,3) = sin(t*omega*1.2+s3)+1.21;
    case 2
        w(:,1) = sin(t*omega+s1)+1.21;
        w(:,2) = sin(t*omega*0.125+s2)+1; % 0.99;   %1.01;
        w(:,3) = sin(t*omega*1.2+s3)+1.21;
    case 3
        w(:,1) = sin(t*1.1*omega+s1)+2;
        %w(:,2) = sin(t*omega*0.125+s2)+1.01;
        w(:,2) = abs(sin(t*omega*0.125-s2/10)+(t.^2)/(3000^2)-0.15);
        w(:,3) = sin(t*omega*1.2+s3)+1;    
    otherwise
        error('Illegal Ind Value!');
end
    
ww = sum(w')';
for j=1:3
    w(:,j)  = w(:,j) ./ ww;
end
PPP = w; 
if false
    figure;
    subplot(2,1,1);
    plot(w,'LineWidth',2);
    ylim([0 1]);
    title(['Mode ' num2str(ind) '. \omega = ' num2str(omega)]);
    subplot(2,1,2);
    area(PPP);
    ylim([0 1]);
    q=1;
end

function [r] = draw_bgr_lines(Z,LW)
if nargin==1
    LW=1;
end
[K,N] =size(Z);
xx = (1:N);
plot(xx,Z(1,:)','-b',xx,Z(2,:)','-g',xx,Z(3,:)','-r','LineWidth',LW); 
r=1;
    
