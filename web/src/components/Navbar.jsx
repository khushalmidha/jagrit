import React, { useContext } from 'react';
import { Link } from 'react-router-dom';
import { AuthContext } from '../context/AuthContext';
import { LanguageContext } from '../context/LanguageContext';
import { User, LogOut } from 'lucide-react';

const Navbar = () => {
  const { user, logout } = useContext(AuthContext);
  const { lang, toggleLanguage, t } = useContext(LanguageContext);

  return (
    <nav className="bg-white border-b border-gray-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex">
            <div className="flex-shrink-0 flex items-center">
              <Link to="/" className="font-serif font-bold text-3xl tracking-tighter text-black">
                Jagrit.
              </Link>
            </div>
            <div className="hidden sm:ml-8 sm:flex sm:space-x-8">
              <Link to="/" className="border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium">
                {t('Top Stories')}
              </Link>
              <Link to="/foryou" className="border-black text-gray-900 inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium">
                {t('For You')}
              </Link>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <button 
              onClick={toggleLanguage}
              className="px-3 py-1 bg-gray-100 text-gray-800 text-xs font-bold uppercase rounded-full hover:bg-gray-200 transition"
            >
              {lang === 'en' ? 'A/अ (HI)' : 'A/अ (EN)'}
            </button>
            
            {user ? (
              <div className="flex items-center space-x-3">
                <Link to="/preferences" className="text-gray-500 hover:text-gray-900" title={t('Preferences')}>
                  <User size={20} />
                </Link>
                <button onClick={logout} className="text-gray-500 hover:text-gray-900" title={t('Logout')}>
                  <LogOut size={20} />
                </button>
              </div>
            ) : (
              <Link to="/login" className="text-sm font-medium text-gray-700 hover:text-gray-900">
                {t('Login')}
              </Link>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
