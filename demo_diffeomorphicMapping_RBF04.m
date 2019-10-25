function demo_diffeomorphicMapping_RBF04
% Imitation using task-parameterized diffeomorphic mapping (for online imitation tasks involving objects)
%
% Sylvain Calinon, 2019

%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbVar = 2; %Dimension of datapoints
nbData = 100; %Number of datapoints in demonstration
nbGrid = 30; %Size of grid
rFactor = 2E-2; %Regularization factor



rg = linspace(-.2,1.2,nbGrid); %Grid range
[xm, ym] = ndgrid(rg,rg);
xg = [xm(:)'; ym(:)']; %Grid datapoints

% dxk = [.1 -.2 .1; .2 -.1 -.2]; %Dispacement
xk = [.1 .8 .4 .1; .1 .2 .9 .8]; %Keypoints x
nbPts = size(xk,2); %Number of landmarks

dxk = [.1 -.2 .1 0; .2 -.1 -.2 .1]; %Dispacement
% dxk = [.1 0 0 0; .2 0 0 0 ]; %Dispacement
xk2 = xk + dxk; %Keypoints Agent 2
x = spline(1:nbPts, xk, linspace(1,nbPts,nbData)); %  x

% horizontal line
nbData_line = 200; % precision of the grid line
for i = 1 : nbGrid
    tmp_x = [ rg; rg(i)*ones(1,nbGrid)];
    xgl(:,:,i) = spline(1:nbGrid,tmp_x, linspace(1,nbGrid,nbData_line));
end
% vertical line
for i = 1 : nbGrid
    tmp_x = [ rg(i)*ones(1,nbGrid); rg];
    xgl(:,:,i+nbGrid) = spline(1:nbGrid,tmp_x, linspace(1,nbGrid,nbData_line));
end

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

%% Task-parameterized diffeomorphic mapping
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%Compute grid (for visualization)
xi = zeros(nbVar, nbGrid^nbVar, nbStates);
h = zeros(nbStates, nbGrid^nbVar);
for i=1:nbStates
	h(i,:) = gaussPDF(xg, Mu(:,i), Sigma(:,:,i));
	xTmp = A(:,:,i) \ (xg - repmat(b(:,i), 1, nbGrid^nbVar));
	xi(:,:,i) = A2(:,:,i) * xTmp + repmat(b2(:,i), 1, nbGrid^nbVar);
end
h = h ./ repmat(sum(h), nbStates, 1);
% h = h ./ repmat(max(h,[],2), 1, nbGrid^nbVar);

xg2 = zeros(nbVar, nbGrid^nbVar);
for i=1:nbStates
	xg2 = xg2 + xi(:,:,i) * diag(h(i,:));
end

%Compute grid line (for validation of diffeomorphic mapping)
for j = 1:nbGrid*2
    xi = zeros(nbVar, nbData_line, nbStates);
    h = zeros(nbStates, nbData_line);
    for i=1:nbStates
        h(i,:) = gaussPDF(xgl(:,:,j), Mu(:,i), Sigma(:,:,i));
        xTmp = A(:,:,i) \ (xgl(:,:,j) - repmat(b(:,i), 1, nbData_line));
        xi(:,:,i) = A2(:,:,i) * xTmp + repmat(b2(:,i), 1, nbData_line);
    end
    h = h ./ repmat(sum(h), nbStates, 1);
%     h = h ./ repmat(max(h,[],2), 1, nbData_line);
    
    xl2 = zeros(nbVar, nbData_line);
    for i=1:nbStates
        xl2 = xl2 + xi(:,:,i) * diag(h(i,:));
    end
    xgl2(:,:,j) = xl2;
end

%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,1500,900]); 
clrmap = lines(nbPts);

%AGENT 1
subplot(1,2,1); hold on; axis off;
plot(xg(1,:), xg(2,:), '.','markersize',8,'color',[.6 .6 .6]);
% plot grid line 1
for i = 1:nbGrid*2
   plot(xgl(1,:,i),xgl(2,:,i),'color',[.6 .6 .6]); 
end
for i=1:nbStates
	plotGMM(Mu(:,i), Sigma(:,:,i), mean(clrmap(id(i,:),:)),.3);
end
for i=1:nbStates
	plot2DArrow(b(:,i), A(:,1,i), mean(clrmap(id(i,:),:)), 2, 1E-2);
	plot2DArrow(b(:,i), A(:,2,i), mean(clrmap(id(i,:),:)), 2, 1E-2);
end
for i=1:nbPts
	plot(xk(1,i), xk(2,i), '.','markersize',40,'color',clrmap(i,:));
end
plot(x(1,:), x(2,:), '-','linewidth',3,'color',[0 0 0]);
axis equal; axis([-.1,1.1,-.1,1.1]);

%AGENT 2
subplot(1,2,2); hold on; axis off;
plot(xg2(1,:), xg2(2,:), '.','markersize',8,'color',[.6 .6 .6]);
% plot grid line 2
for i = 1:nbGrid*2
   plot(xgl2(1,:,i),xgl2(2,:,i),'color',[.6 .6 .6]); 
end
for i=1:nbStates
	plotGMM(Mu2(:,i), Sigma2(:,:,i), mean(clrmap(id(i,:),:)),.3);
end
for i=1:nbStates
	plot2DArrow(b2(:,i), A2(:,1,i), mean(clrmap(id(i,:),:)), 2, 1E-2);
	plot2DArrow(b2(:,i), A2(:,2,i), mean(clrmap(id(i,:),:)), 2, 1E-2);
end
for i=1:nbPts
	plot(xk2(1,i), xk2(2,i), '.','markersize',40,'color',clrmap(i,:));
end
plot(x2(1,:), x2(2,:), '-','linewidth',3,'color',[0 0 0]);
axis equal; axis([-.1,1.1,-.1,1.1]);

pause;
close all;