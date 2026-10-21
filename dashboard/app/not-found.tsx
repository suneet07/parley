import Link from "next/link";
import { Empty } from "@/components/States";

export default function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center p-12">
      <Empty>Page not found.</Empty>
      <div className="mt-4">
        <Link href="/" className="btn">
          Back to overview
        </Link>
      </div>
    </div>
  );
}
