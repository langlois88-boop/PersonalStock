export const Skeleton = ({ className = '' }) => (
  <div className={`animate-pulse bg-slate-800/60 rounded ${className}`} />
);

export const SkeletonCard = () => (
  <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-5 space-y-4">
    <div className="flex justify-between items-center">
      <Skeleton className="h-4 w-24" />
      <Skeleton className="h-6 w-16 rounded-full" />
    </div>
    <div className="space-y-2">
      <Skeleton className="h-3 w-full" />
      <Skeleton className="h-3 w-3/4" />
    </div>
    <div className="pt-2 flex gap-2">
      <Skeleton className="h-8 flex-1 rounded-lg" />
      <Skeleton className="h-8 flex-1 rounded-lg" />
    </div>
  </div>
);
