%% combine Fast diffeomorphic matching and Gaussian RBF mapping
% Xiao Gao  Oct. 2019

clc;close all;clear;
%% Parameters
nbVar = 2; %Dimension of datapoints
nbData = 100; %Number of datapoints in demonstration
nbGrid = 30; %Size of grid
rFactor = 50E-2; %Regularization factor

paraNb = 150; % iteration times K
paraMu = 0.9;
paraBelta = 0.5;

rg = linspace(-3.5,3.5,nbGrid); %Grid range
[xm, ym] = ndgrid(rg,rg);
xg = [xm(:)'; ym(:)']; %Grid datapoints

% dxk = [.1 -.2 .1; .2 -.1 -.2]; %Dispacement
xk = [-3 -1.5 -0.5 2;1 -2 1.5 0];  % object position
nbPts = size(xk,2); %Number of landmarks
dxk = [0 0.5 1 0.5; 1 -0.5 1 -1.5]; %Dispacement
xk2 = xk + dxk; %Keypoints Agent 2
x = spline(1:nbPts, xk, linspace(1,nbPts,nbData)); %Motion of Agent 1
% y = x(:,1) + ( (1:nbData) -1) ./(nbData-1) .* ( x(:,end ) - x(:,1)); % y is the line

% horizontal line
nbData_line = 1000; % precision of the grid line
for i = 1 : nbGrid
    xgl(:,:,i) = [linspace(min(rg),max(rg),nbData_line);rg(i)*ones(1,nbData_line)];
end
% vertical line
for i = 1 : nbGrid
    xgl(:,:,i+nbGrid) = [rg(i)*ones(1,nbData_line);linspace(min(rg),max(rg),nbData_line)];
end
%% calculate Gaussian distribution
id = nchoosek(1:nbPts,2); %List all combinations of two points
nbStates = size(id,1); %Number of Gaussians
R = [cos(-pi/2) -sin(-pi/2); sin(-pi/2) cos(-pi/2)]; %Rotation operator
for i=1:nbStates
	%Agent 1
	Mu(:,i) = mean(xk(:,id(i,:)),2); %Center
	Sigma(:,:,i) = cov(xk(:,id(i,:))') + eye(nbVar) .* rFactor; %Covariance
	aTmp = xk(:,id(i,2)) - xk(:,id(i,1)); 
	A(:,:,i) = [R*aTmp.*rFactor.^.5./norm(aTmp), aTmp]; %Coordinate system
	b(:,i) = xk(:,id(i,1)); %Origin
	%Agent 2
	Mu2(:,i) = mean(xk2(:,id(i,:)),2); %Center
	Sigma2(:,:,i) = cov(xk2(:,id(i,:))') + eye(nbVar) .* rFactor; %Covariance
	aTmp = xk2(:,id(i,2)) - xk2(:,id(i,1)); 
	A2(:,:,i) = [R*aTmp.*rFactor.^.5./norm(aTmp), aTmp]; %Coordinate system
	b2(:,i) = xk2(:,id(i,1)); %Origin
end
%% Task-parameterized Gaussian mapping
xi = zeros(nbVar, nbData, nbStates);
h = zeros(nbStates, nbData);
for i=1:nbStates
	h(i,:) = gaussPDF(x, Mu(:,i), Sigma(:,:,i));
	xTmp = A(:,:,i) \ (x - repmat(b(:,i), 1, nbData));
	xi(:,:,i) = A2(:,:,i) * xTmp + repmat(b2(:,i), 1, nbData);
end
h = h ./ repmat(sum(h), nbStates, 1);
% h = h ./ repmat(max(h,[],2), 1, nbData);

x2 = zeros(nbVar, nbData);
for i=1:nbStates
	x2 = x2 + xi(:,:,i) * diag(h(i,:));
end

%% iteration algorithm for diffeomorphic mapping
z = x;
rho_ = zeros(1,paraNb);
options = optimoptions('fmincon','Display','off');
for i = 1: paraNb
    [~,m] = max( sum((z - x2).^2) );
    p(:,i) = z(:,m);
    q(:,i) = x2(:,m);
    v(:,i) = paraBelta * (q(:,i) - p(:,i));
    up_bound = sqrt(exp(1)/2)/norm(v(:,i),2);
    % solve min rho
    %     Phi = @(rho) z + v(:,i).* exp(-rho^2 * sum((z - p(:,i)).^2));  % transfer z with function Phi
    dis = @(rho) sum(sum((z + v(:,i).* exp(-rho^2 * sum((z - p(:,i)).^2))  - x2).^2))/nbData;   % distance, object function
    [rho_(i),dis_(i)] = fmincon(dis,0.1,[],[],[],[],0,up_bound,[],options);    % solve the minimum
    z = z + v(:,i).* exp(-rho_(i)^2 * sum((z - p(:,i)).^2));  % update z to phi(z)
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
plot(x(1,:),x(2,:),'LineWidth',1.5,'color','k');  % spline line  x
for i=1:nbStates
	plotGMM(Mu(:,i), Sigma(:,:,i), mean(clrmap(id(i,:),:)),.3);
end
for i=1:nbPts
    plot(xk(1,i), xk(2,i), '.','markersize',40,'color',clrmap(i,:));
end
for i = 1:nbGrid*2
    plot(xgl(1,:,i),xgl(2,:,i),'color',[.6 .6 .6]); % plot grid line 1
end
axis equal; axis([-3.5,3.5,-3.5,3.5]);

% mapping
subplot(1,2,2);hold on; axis off;
plot(x2(1,:),x2(2,:),'--','LineWidth',1.5);  % spline line  x
for i=1:nbPts
    plot(xk2(1,i), xk2(2,i), '.','markersize',40,'color',clrmap(i,:));
end
for i=1:nbStates
	plotGMM(Mu2(:,i), Sigma2(:,:,i), mean(clrmap(id(i,:),:)),.3);
end
plot(z(1,:),z(2,:),'-','LineWidth',1);  %  z = Phi(y)
for i = 1:nbGrid*2
    plot(xgl2(1,:,i),xgl2(2,:,i),'color',[.6 .6 .6]); % plot grid line 1
end
axis equal; axis([-3.5,3.5,-3.5,3.5]);


