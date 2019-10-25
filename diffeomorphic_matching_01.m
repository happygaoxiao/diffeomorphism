%% code for reproducing the algorithm of the paper 'Fast diffeomorphic matching'
% Xiao Gao  Oct. 2019

clc;close all;clear;

%% Parameters
nbVar = 2; %Dimension of datapoints

nbData = 100; %Number of datapoints in demonstration
nbGrid = 30; %Size of grid

paraNb = 150; % iteration times K
paraMu = 0.9;
paraBelta = 0.5;

rg = linspace(-3.5,3.5,nbGrid); %Grid range
[xm, ym] = ndgrid(rg,rg);
xg = [xm(:)'; ym(:)']; %Grid datapoints

% dxk = [.1 -.2 .1; .2 -.1 -.2]; %Dispacement
% xk = [.1 .8 .4 0; .1 .2 .9 0.9]; %Keypoints Agent 1
xk = [-3 -2.5 -1.5 -0.5 2;1 2 -2 1.5 0];
nbPts = size(xk,2); %Number of landmarks
dxk = [.1 -.2 .1 0; .2 -.1 -.2 .1]; %Dispacement
% dxk = [.1 0 0 0; .2 0 0 0 ]; %Dispacement
% xk2 = xk + dxk; %Keypoints Agent 2
x = spline(1:nbPts, xk, linspace(1,nbPts,nbData)); %Motion of Agent 1
y = x(:,1) + ( (1:nbData) -1) ./(nbData-1) .* ( x(:,end ) - x(:,1)); % y is the line


% horizontal line
nbData_line = 200; % precision of the grid line
for i = 1 : nbGrid
    xgl(:,:,i) = [linspace(min(rg),max(rg),nbData_line);rg(i)*ones(1,nbData_line)];
end
% vertical line
for i = 1 : nbGrid
    xgl(:,:,i+nbGrid) = [rg(i)*ones(1,nbData_line);linspace(min(rg),max(rg),nbData_line)];
end

%% iteration algorithm for diffeomorphic mapping
z = y;
rho_ = zeros(1,paraNb);
% figure('position',[-1919 11 1920 962]); clrmap = lines(nbPts);
% hold on; axis off;
% plot(x(1,:),x(2,:),'--','LineWidth',1.5);  % spline line  x
% for i=1:nbPts
%     plot(xk(1,i), xk(2,i), '.','markersize',40,'color',clrmap(i,:));
% end
% plot(y(1,:),y(2,:),'LineWidth',1.5);     % straight line y
options = optimoptions('fmincon','Display','off');
for i = 1: paraNb
    [~,m] = max( sum((z - x).^2) );
    p(:,i) = z(:,m);
    q(:,i) = x(:,m);
    v(:,i) = paraBelta * (q(:,i) - p(:,i));
    up_bound = sqrt(exp(1)/2)/norm(v(:,i),2);
    % solve min rho
    %     Phi = @(rho) z + v(:,i).* exp(-rho^2 * sum((z - p(:,i)).^2));  % transfer z with function Phi
    dis = @(rho) sum(sum((z + v(:,i).* exp(-rho^2 * sum((z - p(:,i)).^2))  - x).^2))/nbData;   % object function
    [rho_(i),dis_(i)] = fmincon(dis,0.1,[],[],[],[],0,up_bound,[],options);            % solve the minimum of distance
    z = z + v(:,i).* exp(-rho_(i)^2 * sum((z - p(:,i)).^2));  % update z to phi(z)    
    % plot(z(1,:),z(2,:),'-.','LineWidth',1.5);  % spline line  x
    % pause(0.1)                                 % pause to show the iteration process
end

% grid line mapping
for j = 1:nbGrid*2
    z1 = xgl(:,:,j);
    for i = 1: paraNb
        z1 = z1 + v(:,i).* exp(-rho_(i)^2 * sum((z1 - p(:,i)).^2));  
    end
    xgl2(:,:,j) = z1;   % new grid line
end

%% plots
figure('position',[10,10,1800,900]); clrmap = lines(nbPts);
% original
subplot(1,2,1);hold on; axis off;
plot(x(1,:),x(2,:),'--','LineWidth',1.5);  % spline line  x
for i=1:nbPts
    plot(xk(1,i), xk(2,i), '.','markersize',30,'color',clrmap(i,:));
end
plot(y(1,:),y(2,:),'LineWidth',1.5);     % straight line y
for i = 1:nbGrid*2
    plot(xgl(1,:,i),xgl(2,:,i),'color',[.6 .6 .6]); % plot grid line 1
end
axis equal; axis([-3.5,3.5,-3.5,3.5]);

% mapping
subplot(1,2,2);hold on; axis off;
plot(x(1,:),x(2,:),'--','LineWidth',1.5);  % spline line  x
plot(z(1,:),z(2,:),'-','LineWidth',1);  %  z = Phi(y)
for i = 1:nbGrid*2
    plot(xgl2(1,:,i),xgl2(2,:,i),'color',[.6 .6 .6]); % plot grid line 1
end
axis equal; axis([-3.5,3.5,-3.5,3.5]);


