import React from 'react';

const FeedSkeleton = () => {
  return (
    <div className="magazine-grid">
      <div className="hero-card flex flex-col md:flex-row bg-white rounded-md border border-gray-100 animate-pulse">
        <div className="md:w-2/3 h-64 bg-gray-200"></div>
        <div className="md:w-1/3 p-5 flex flex-col justify-center space-y-3">
          <div className="h-4 bg-gray-200 w-1/4 rounded"></div>
          <div className="h-8 bg-gray-200 w-full rounded"></div>
          <div className="h-8 bg-gray-200 w-5/6 rounded"></div>
          <div className="h-4 bg-gray-200 w-full rounded mt-4"></div>
          <div className="h-4 bg-gray-200 w-2/3 rounded"></div>
        </div>
      </div>
      
      {[1, 2, 3, 4, 5, 6].map(i => (
        <div key={i} className="flex flex-col bg-white rounded-md border border-gray-100 animate-pulse">
          <div className="h-48 bg-gray-200"></div>
          <div className="p-5 space-y-3">
            <div className="h-4 bg-gray-200 w-1/4 rounded"></div>
            <div className="h-6 bg-gray-200 w-full rounded"></div>
            <div className="h-6 bg-gray-200 w-5/6 rounded"></div>
            <div className="h-4 bg-gray-200 w-full rounded mt-2"></div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default FeedSkeleton;
